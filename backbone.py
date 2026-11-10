from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear
import torch.nn as nn
from collections import OrderedDict

import torch
import torch.nn.functional as F



#meta lora




class MetaSRLoRAConv2d(MetaConv2d):
    """
    Meta-compatible SRLoRA convolution.

    The effective convolution weight is:

        W_effective =
            W_base
            + scaling * (B_shared @ A_shared)
            + scaling * (B_rotating @ A_rotating)

    The shared and rotating ranks add up to the total active rank.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        shared_rank=10,
        rotating_rank=6,
        lora_alpha=16.0,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )

        if shared_rank < 0:
            raise ValueError("`shared_rank` must be non-negative, got {}.".format(shared_rank))

        if rotating_rank < 0:
            raise ValueError("`rotating_rank` must be non-negative, got {}.".format(rotating_rank))

        self.shared_rank = int(shared_rank)
        self.rotating_rank = int(rotating_rank)
        self.total_rank = self.shared_rank + self.rotating_rank
        self.lora_alpha = float(lora_alpha)

        if self.total_rank > 0 and self.lora_alpha <= 0:
            raise ValueError("`lora_alpha` must be positive when the total rank is positive.")

        kernel_height, kernel_width = self.kernel_size
        self.flattened_input_dimension = (self.in_channels // self.groups) * kernel_height * kernel_width
        self.maximum_rank = min(self.out_channels, self.flattened_input_dimension)

        if self.total_rank > self.maximum_rank:
            raise ValueError(
                "Total rank {} exceeds maximum possible rank {}.".format(
                    self.total_rank,
                    self.maximum_rank
                )
            )

        if self.total_rank > 0:
            self.scaling = self.lora_alpha / float(self.total_rank)
        else:
            self.scaling = 0.0

        if self.shared_rank > 0:
            self.lora_shared_A = nn.Parameter(
                torch.empty(
                    self.shared_rank,
                    self.flattened_input_dimension
                )
            )

            self.lora_shared_B = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.shared_rank
                )
            )
        else:
            self.register_parameter("lora_shared_A", None)
            self.register_parameter("lora_shared_B", None)

        if self.rotating_rank > 0:
            self.lora_rotating_A = nn.Parameter(
                torch.empty(
                    self.rotating_rank,
                    self.flattened_input_dimension
                )
            )

            self.lora_rotating_B = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.rotating_rank
                )
            )
        else:
            self.register_parameter("lora_rotating_A", None)
            self.register_parameter("lora_rotating_B", None)

        self.register_buffer(
            "initial_singular_values",
            torch.empty(self.maximum_rank)
        )        

        remaining_rank = (
            self.maximum_rank
            - self.total_rank
        )

        self.register_buffer(
            "unused_U",
            torch.empty(
                self.out_channels,
                remaining_rank
            )
        )

        self.register_buffer(
            "unused_S",
            torch.empty(
                remaining_rank
            )
        )

        self.register_buffer(
            "unused_V",
            torch.empty(
                self.flattened_input_dimension,
                remaining_rank
            )
        )

        self.register_buffer(
            "next_pool_index",
            torch.zeros(
                (),
                dtype=torch.long
            )
        )

        self.register_buffer(
            "warmup_finalized",
            torch.zeros(
                (),
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "rotation_count",
            torch.zeros(
                (),
                dtype=torch.long
            )
        )

        if self.total_rank > 0:
            self.reset_srlora_parameters()
        # Autograd remains enabled for full-gradient diagnostics.
        # This weight will later be excluded from both the inner update
        # and the outer optimizer, so it remains frozen during training.
        self.weight.requires_grad_(False)

        if self.bias is not None:
            self.bias.requires_grad_(False)

    def reset_srlora_parameters(self):
        """
        Initial SRLoRA decomposition:

            W_original =
                W_base
                + scaling * B_shared A_shared
                + scaling * B_rotating A_rotating

        The first `shared_rank` singular directions initialize
        the shared branch.

        The following `rotating_rank` singular directions initialize
        the rotating branch.
        """

        with torch.no_grad():
            original_weight = (
                self.weight.detach()
                .reshape(self.out_channels, -1)
                .clone()
            )

            U, S, V = torch.svd(
                original_weight,
                some=True
            )

            self.initial_singular_values.copy_(
                S.detach().to(
                    dtype=self.initial_singular_values.dtype
                )
            )




            pool_start = self.total_rank

            self.unused_U.copy_(
                U[:, pool_start:]
            )

            self.unused_S.copy_(
                S[pool_start:]
            )

            self.unused_V.copy_(
                V[:, pool_start:]
            )

            self.next_pool_index.zero_()
            self.warmup_finalized.zero_()
            self.rotation_count.zero_()

            shared_start = 0
            shared_end = self.shared_rank
            rotating_start = shared_end
            rotating_end = shared_end + self.rotating_rank

            principal_weight = torch.zeros_like(
                original_weight
            )

            if self.shared_rank > 0:
                U_shared = U[:, shared_start:shared_end]
                S_shared = S[shared_start:shared_end]
                V_shared = V[:, shared_start:shared_end]

                shared_sqrt_values = torch.sqrt(
                    S_shared / self.scaling
                )

                shared_B_init = (
                    U_shared
                    * shared_sqrt_values.unsqueeze(0)
                )

                shared_A_init = (
                    shared_sqrt_values.unsqueeze(1)
                    * V_shared.t()
                )

                self.lora_shared_A.copy_(
                    shared_A_init
                )

                self.lora_shared_B.copy_(
                    shared_B_init
                )

                principal_weight.add_(
                    self.scaling
                    * (
                        shared_B_init
                        @ shared_A_init
                    )
                )

            if self.rotating_rank > 0:
                U_rotating = U[:, rotating_start:rotating_end]
                S_rotating = S[rotating_start:rotating_end]
                V_rotating = V[:, rotating_start:rotating_end]

                rotating_sqrt_values = torch.sqrt(
                    S_rotating / self.scaling
                )

                rotating_B_init = (
                    U_rotating
                    * rotating_sqrt_values.unsqueeze(0)
                )

                rotating_A_init = (
                    rotating_sqrt_values.unsqueeze(1)
                    * V_rotating.t()
                )

                self.lora_rotating_A.copy_(
                    rotating_A_init
                )

                self.lora_rotating_B.copy_(
                    rotating_B_init
                )

                principal_weight.add_(
                    self.scaling
                    * (
                        rotating_B_init
                        @ rotating_A_init
                    )
                )

            residual_weight = (
                original_weight
                - principal_weight
            )

            self.weight.copy_(
                residual_weight.view_as(
                    self.weight
                )
            )

            reconstruction = (
                self.weight
                .reshape(self.out_channels, -1)
                .clone()
            )

            if self.shared_rank > 0:
                reconstruction.add_(
                    self.scaling
                    * (
                        self.lora_shared_B
                        @ self.lora_shared_A
                    )
                )

            if self.rotating_rank > 0:
                reconstruction.add_(
                    self.scaling
                    * (
                        self.lora_rotating_B
                        @ self.lora_rotating_A
                    )
                )

            reconstruction_error = (
                reconstruction
                - original_weight
            ).abs().max().item()

            if reconstruction_error > 1e-5:
                raise RuntimeError(
                    "SRLoRA initialization reconstruction error is too large: {:.6e}".format(
                        reconstruction_error
                    )
                )

#BA ro misaze
    def get_shared_matrix(self):
        if self.shared_rank == 0:
            return self.weight.new_zeros(
                self.out_channels,
                self.flattened_input_dimension
            )

        return (
            self.scaling
            * (
                self.lora_shared_B
                @ self.lora_shared_A
            )
        )
#BA_r ro misaze
    def get_rotating_matrix(self):
        if self.rotating_rank == 0:
            return self.weight.new_zeros(
                self.out_channels,
                self.flattened_input_dimension
            )

        return (
            self.scaling
            * (
                self.lora_rotating_B
                @ self.lora_rotating_A
            )
        )
# w + BA_s + BA_r
    def get_effective_weight_matrix(self):
        base_weight = self.weight.reshape(
            self.out_channels,
            -1
        )

        return (
            base_weight
            + self.get_shared_matrix()
            + self.get_rotating_matrix()
        )
#usv  ro bedi, A and B ro misaze
    def build_factors_from_svd(
        self,
        U,
        S,
        V
    ):
        """
        Build B and A such that:

            scaling * B A
                = U diag(S) V^T
        """

        if S.numel() == 0:
            B = U.new_empty(
                self.out_channels,
                0
            )

            A = V.new_empty(
                0,
                self.flattened_input_dimension
            )

            return B, A

        sqrt_values = torch.sqrt(
            S.clamp_min(0.0)
            / self.scaling
        )

        B = (
            U
            * sqrt_values.unsqueeze(0)
        )

        A = (
            sqrt_values.unsqueeze(1)
            * V.t()
        )

        return B, A


    def has_remaining_directions(self):
        current_index = int(
            self.next_pool_index.item()
        )

        return (
            current_index
            < self.unused_S.numel()
        )

    def remaining_direction_count(self):
        current_index = int(
            self.next_pool_index.item()
        )

        return max(
            0,
            self.unused_S.numel()
            - current_index
        )




    @torch.no_grad()
    def finalize_warmup(self):
        """
        Re-decompose the learned rank-total active update
        after warm-up.

        The top shared_rank singular components become
        the persistent shared branch.

        The following rotating_rank components become
        the rotating branch.

        The base weight is not changed.
        """

        if bool(
            self.warmup_finalized.item()
        ):
            return {
                "changed": False,
                "reason": "already_finalized"
            }

        effective_before = (
            self.get_effective_weight_matrix()
            .detach()
            .clone()
        )

        active_matrix = (
            self.get_shared_matrix()
            + self.get_rotating_matrix()
        )

        U, S, V = torch.svd(
            active_matrix,
            some=True
        )

        shared_start = 0
        shared_end = self.shared_rank

        rotating_start = shared_end
        rotating_end = (
            shared_end
            + self.rotating_rank
        )

        if self.shared_rank > 0:
            shared_B, shared_A = (
                self.build_factors_from_svd(
                    U[:, shared_start:shared_end],
                    S[shared_start:shared_end],
                    V[:, shared_start:shared_end]
                )
            )

            self.lora_shared_B.copy_(
                shared_B
            )

            self.lora_shared_A.copy_(
                shared_A
            )

        if self.rotating_rank > 0:
            rotating_B, rotating_A = (
                self.build_factors_from_svd(
                    U[:, rotating_start:rotating_end],
                    S[rotating_start:rotating_end],
                    V[:, rotating_start:rotating_end]
                )
            )

            self.lora_rotating_B.copy_(
                rotating_B
            )

            self.lora_rotating_A.copy_(
                rotating_A
            )


        effective_after_uncorrected = self.get_effective_weight_matrix().detach().clone()
        self.weight.add_((effective_before - effective_after_uncorrected).view_as(self.weight))
        effective_after = self.get_effective_weight_matrix().detach().clone()

        relative_error = (
            (
                effective_after
                - effective_before
            ).norm()
            / (
                effective_before.norm()
                + 1e-12
            )
        ).item()

        if relative_error > 1e-5:
            raise RuntimeError(
                "Warm-up finalization changed the effective "
                "weight. Relative error: {:.6e}".format(
                    relative_error
                )
            )

        squared_singular_values = (
            S.square()
        )

        total_energy = (
            squared_singular_values
            .sum()
            .clamp_min(1e-12)
        )

        shared_energy = (
            squared_singular_values[
                :self.shared_rank
            ].sum()
            / total_energy
        ).item()

        if (
            self.shared_rank > 0
            and self.shared_rank
            < S.numel()
        ):
            spectral_gap = (
                (
                    S[self.shared_rank - 1]
                    - S[self.shared_rank]
                )
                / (
                    S[self.shared_rank - 1]
                    + 1e-12
                )
            ).item()
        else:
            spectral_gap = 0.0

        self.warmup_finalized.fill_(
            True
        )

        return {
            "changed": True,
            "reason": "warmup_finalized",
            "relative_weight_error": relative_error,
            "shared_energy_fraction": shared_energy,
            "spectral_gap": spectral_gap
        }


    @torch.no_grad()
    def rotate_subspace(self):
        """
        Replace rotating directions with unused
        spectral directions while preserving the
        effective convolution weight exactly.

        For a complete rotation:

            base_new =
                base_old
                + rotating_old
                - rotating_new

        For a final partial rotation, the strongest
        old rotating components are retained.
        """

        if self.rotating_rank == 0:
            return {
                "changed": False,
                "reason": "zero_rotating_rank"
            }

        if not bool(
            self.warmup_finalized.item()
        ):
            raise RuntimeError(
                "Cannot rotate SRLoRA directions before "
                "warm-up finalization."
            )

        remaining_count = (
            self.remaining_direction_count()
        )

        if remaining_count == 0:
            return {
                "changed": False,
                "reason": "pool_exhausted",
                "remaining_pool": 0,
                 "rotation_count": int(self.rotation_count.item())
            }

        effective_before = (
            self.get_effective_weight_matrix()
            .detach()
            .clone()
        )

        old_rotating_matrix = (
            self.get_rotating_matrix()
            .detach()
            .clone()
        )

        replace_count = min(
            self.rotating_rank,
            remaining_count
        )

        retain_count = (
            self.rotating_rank
            - replace_count
        )

        if retain_count > 0:
            old_U, old_S, old_V = torch.svd(
                old_rotating_matrix.contiguous(),
                some=True
            )

            retained_U = old_U[:, :retain_count].contiguous()
            retained_S = old_S[:retain_count].contiguous()
            retained_V = old_V[:, :retain_count].contiguous()

            retained_matrix = torch.mm(
                (retained_U * retained_S.unsqueeze(0)).contiguous(),
                retained_V.t().contiguous()
            )

            retained_B, retained_A = self.build_factors_from_svd(
                retained_U,
                retained_S,
                retained_V
            )

            
        else:
            retained_matrix = (
                old_rotating_matrix.new_zeros(
                    old_rotating_matrix.shape
                )
            )

            retained_B = (
                old_rotating_matrix.new_empty(
                    self.out_channels,
                    0
                )
            )

            retained_A = (
                old_rotating_matrix.new_empty(
                    0,
                    self.flattened_input_dimension
                )
            )

        discarded_matrix = (
            old_rotating_matrix
            - retained_matrix
        )

        pool_start = int(
            self.next_pool_index.item()
        )

        pool_end = (
            pool_start
            + replace_count
        )

        new_U = self.unused_U[:, pool_start:pool_end].contiguous()
        new_S = self.unused_S[pool_start:pool_end].contiguous()
        new_V = self.unused_V[:, pool_start:pool_end].contiguous()

        new_pool_matrix = torch.mm(
            (new_U * new_S.unsqueeze(0)).contiguous(),
            new_V.t().contiguous()
        )



        new_B, new_A = (
            self.build_factors_from_svd(
                new_U,
                new_S,
                new_V
            )
        )

        rotating_B_new = torch.cat(
            [
                retained_B,
                new_B
            ],
            dim=1
        )

        rotating_A_new = torch.cat(
            [
                retained_A,
                new_A
            ],
            dim=0
        )

        if rotating_B_new.shape[1] != self.rotating_rank:
            raise RuntimeError(
                "The new rotating B factor has an invalid rank."
            )

        if rotating_A_new.shape[0] != self.rotating_rank:
            raise RuntimeError(
                "The new rotating A factor has an invalid rank."
            )

        base_matrix = (
            self.weight
            .reshape(
                self.out_channels,
                -1
            )
            .detach()
            .clone()
        )

        base_matrix_new = (
            base_matrix
            + discarded_matrix
            - new_pool_matrix
        )

        self.weight.copy_(
            base_matrix_new.view_as(
                self.weight
            )
        )

        self.lora_rotating_B.copy_(
            rotating_B_new
        )

        self.lora_rotating_A.copy_(
            rotating_A_new
        )


        effective_after_uncorrected = self.get_effective_weight_matrix().detach().clone()
        self.weight.add_((effective_before - effective_after_uncorrected).view_as(self.weight))

        self.next_pool_index.fill_(
            pool_end
        )

        self.rotation_count.add_(
            1
        )

        effective_after = (
            self.get_effective_weight_matrix()
            .detach()
            .clone()
        )

        relative_error = (
            (
                effective_after
                - effective_before
            ).norm()
            / (
                effective_before.norm()
                + 1e-12
            )
        ).item()

        if relative_error > 1e-5:
            raise RuntimeError(
                "SRLoRA rotation changed the effective "
                "weight. Relative error: {:.6e}".format(
                    relative_error
                )
            )

        remaining_after = (
            self.remaining_direction_count()
        )

        if retain_count > 0:
            rotation_type = (
                "partial_rotation"
            )
        else:
            rotation_type = (
                "full_rotation"
            )

        return {
            "changed": True,
            "reason": rotation_type,
            "replace_count": replace_count,
            "retain_count": retain_count,
            "remaining_pool": remaining_after,
            "rotation_count": int(
                self.rotation_count.item()
            ),
            "relative_weight_error": relative_error
        }




    def get_initial_spectral_energy_info(
        self,
        thresholds=(0.50, 0.70, 0.80, 0.90)
    ):


        singular_values = (
            self.initial_singular_values
        )

        squared_singular_values = (
            singular_values.pow(2)
        )

        total_energy = (
            squared_singular_values
            .sum()
            .clamp_min(1e-12)
        )

        cumulative_energy = (
            torch.cumsum(
                squared_singular_values,
                dim=0
            )
            / total_energy
        )

        rank_for_threshold = {}

        for threshold in thresholds:
            threshold_tensor = cumulative_energy.new_tensor(
                threshold
            )

            threshold_index = torch.searchsorted(
                cumulative_energy,
                threshold_tensor
            )

            rank_for_threshold[threshold] = min(
                int(threshold_index.item()) + 1,
                singular_values.numel()
            )

        available_shared_rank = min(
            self.shared_rank,
            singular_values.numel()
        )

        available_total_rank = min(
            self.total_rank,
            singular_values.numel()
        )

        shared_rank_energy = (
            cumulative_energy[
                available_shared_rank - 1
            ].item()
            if available_shared_rank > 0
            else 0.0
        )

        total_rank_energy = (
            cumulative_energy[
                available_total_rank - 1
            ].item()
            if available_total_rank > 0
            else 0.0
        )

        rotating_rank_energy = max(
            0.0,
            total_rank_energy
            - shared_rank_energy
        )

        return {
            "original_shape": (
                self.out_channels,
                self.flattened_input_dimension
            ),
            "maximum_rank": singular_values.numel(),
            "shared_rank": self.shared_rank,
            "rotating_rank": self.rotating_rank,
            "total_rank": self.total_rank,
            "shared_rank_energy": shared_rank_energy,
            "rotating_rank_energy": rotating_rank_energy,
            "total_rank_energy": total_rank_energy,
            "rank_for_threshold": rank_for_threshold,
            "singular_values": singular_values.clone()
        }
     # original lora computation: a*b
    def forward(self, inputs, params=None):
        if params is None:
            params = OrderedDict(
                self.named_parameters()
            )

        weight = params.get(
            "weight",
            self.weight
        )

        bias = params.get(
            "bias",
            self.bias
        )

        effective_weight = weight

        if self.shared_rank > 0:
            shared_A = params.get(
                "lora_shared_A",
                self.lora_shared_A
            )

            shared_B = params.get(
                "lora_shared_B",
                self.lora_shared_B
            )

            shared_weight = (
                shared_B
                @ shared_A
            ).view_as(weight)

            effective_weight = (
                effective_weight
                + self.scaling
                * shared_weight
            )

        if self.rotating_rank > 0:
            rotating_A = params.get(
                "lora_rotating_A",
                self.lora_rotating_A
            )

            rotating_B = params.get(
                "lora_rotating_B",
                self.lora_rotating_B
            )

            rotating_weight = (
                rotating_B
                @ rotating_A
            ).view_as(weight)

            effective_weight = (
                effective_weight
                + self.scaling
                * rotating_weight
            )

        return F.conv2d(
            inputs,
            effective_weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    # more efiicient lora computation
    def forward_eftezah(self, inputs, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        weight = params.get("weight", self.weight)
        bias = params.get("bias", self.bias)

        # Frozen full-rank base branch
        base_output = F.conv2d(
            inputs,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        if self.total_rank == 0:
            return base_output

        # The current SRLoRA factorization assumes ordinary convolution.
        if self.groups != 1:
            raise NotImplementedError("Factorized SRLoRA forward currently supports groups=1 only.")

        A_parts = []
        B_parts = []

        if self.shared_rank > 0:
            shared_A = params.get("lora_shared_A", self.lora_shared_A)
            shared_B = params.get("lora_shared_B", self.lora_shared_B)
            A_parts.append(shared_A)
            B_parts.append(shared_B)

        if self.rotating_rank > 0:
            rotating_A = params.get("lora_rotating_A", self.lora_rotating_A)
            rotating_B = params.get("lora_rotating_B", self.lora_rotating_B)
            A_parts.append(rotating_A)
            B_parts.append(rotating_B)

        A_total = torch.cat(A_parts, dim=0)
        B_total = torch.cat(B_parts, dim=1)

        active_rank = A_total.shape[0]
        kernel_height, kernel_width = self.kernel_size

        A_kernel = A_total.view(
            active_rank,
            self.in_channels,
            kernel_height,
            kernel_width
        )

        # Low-rank spatial convolution: Cin -> rank
        low_rank_features = F.conv2d(
            inputs,
            A_kernel,
            None,
            self.stride,
            self.padding,
            self.dilation,
            1
        )

        B_kernel = B_total.view(
            self.out_channels,
            active_rank,
            1,
            1
        )

        # 1x1 convolution: rank -> Cout
        low_rank_output = F.conv2d(
            low_rank_features,
            B_kernel,
            None,
            1,
            0,
            1,
            1
        )

        return base_output + self.scaling * low_rank_output    



def conv3x3(
    in_channels,
    out_channels,
    shared_rank=10,
    rotating_rank=6,
    lora_alpha=16.0,
    **kwargs
        ):
    return MetaSequential(
        OrderedDict([
            (
                "conv",
                MetaSRLoRAConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    shared_rank=shared_rank,
                    rotating_rank=rotating_rank,
                    lora_alpha=lora_alpha,
                    **kwargs
                )
            ),
            (
                "norm",
                MetaBatchNorm2d(
                    out_channels,
                    momentum=1.0,
                    track_running_stats=False
                )
            ),
            ("relu", nn.ReLU()),
            ("pool", nn.MaxPool2d(2))
        ])
    )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(
        self,
        in_channels,
        out_features,
        hidden_size=64,
        shared_rank=10,
        rotating_rank=6,
        lora_alpha=16.0,
        first_layer_shared_rank=None,
        first_layer_rotating_rank=None,
        first_layer_lora_alpha=None,
        image_size=84
    ):
        super().__init__()
        first_layer_shared_rank = shared_rank if first_layer_shared_rank is None else first_layer_shared_rank
        first_layer_rotating_rank = rotating_rank if first_layer_rotating_rank is None else first_layer_rotating_rank
        first_layer_lora_alpha = lora_alpha if first_layer_lora_alpha is None else first_layer_lora_alpha
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.image_size = image_size
        
        self.features = MetaSequential(
            OrderedDict([
                ("block1", conv3x3(in_channels, hidden_size, first_layer_shared_rank, first_layer_rotating_rank, first_layer_lora_alpha)),
                ("block2", conv3x3(hidden_size, hidden_size, shared_rank, rotating_rank, lora_alpha)),
                ("block3", conv3x3(hidden_size, hidden_size, shared_rank, rotating_rank, lora_alpha)),
                ("block4", conv3x3(hidden_size, hidden_size, shared_rank, rotating_rank, lora_alpha))
            ])
)
        
        
        
        final_spatial_size = image_size // (2 ** 4)

        classifier_input_size = (
            hidden_size
            * final_spatial_size
            * final_spatial_size
        )

        self.classifier = MetaLinear(
            classifier_input_size,
            out_features
        )


        self.shared_rank = shared_rank
        self.rotating_rank = rotating_rank
        self.total_rank = shared_rank + rotating_rank
        self.lora_alpha = lora_alpha

        self.first_layer_shared_rank = first_layer_shared_rank
        self.first_layer_rotating_rank = first_layer_rotating_rank
        self.first_layer_total_rank = first_layer_shared_rank + first_layer_rotating_rank
        self.first_layer_lora_alpha = first_layer_lora_alpha

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits



    def finalize_srlora_warmup(self):
        reports = []

        for module_name, module in self.named_modules():
            if not isinstance(
                module,
                MetaSRLoRAConv2d
            ):
                continue

            report = module.finalize_warmup()

            report["layer"] = (
                module_name
            )

            reports.append(
                report
            )

        return reports

    def rotate_srlora_subspaces(self):
        reports = []

        for module_name, module in self.named_modules():
            if not isinstance(
                module,
                MetaSRLoRAConv2d
            ):
                continue

            report = (
                module.rotate_subspace()
            )

            report["layer"] = (
                module_name
            )

            reports.append(
                report
            )

        return reports

    def has_remaining_srlora_directions(self):
        for module in self.modules():
            if (
                isinstance(
                    module,
                    MetaSRLoRAConv2d
                )
                and module.has_remaining_directions()
            ):
                return True

        return False





def print_initial_spectral_energy(model):
    print(
        "\n========== Initial Weight Spectral Energy ==========",
        flush=True
    )

    for module_name, module in model.named_modules():
        if not isinstance(
            module,
            MetaSRLoRAConv2d
        ):
            continue

        info = (
            module
            .get_initial_spectral_energy_info()
        )

        ranks = info[
            "rank_for_threshold"
        ]

        print(
            "[{}] original-shape={} / max-rank={}".format(
                module_name,
                info["original_shape"],
                info["maximum_rank"]
            ),
            flush=True
        )

        print(
            "  rank for 50% energy: {}".format(
                ranks[0.50]
            ),
            flush=True
        )

        print(
            "  rank for 70% energy: {}".format(
                ranks[0.70]
            ),
            flush=True
        )

        print(
            "  rank for 80% energy: {}".format(
                ranks[0.80]
            ),
            flush=True
        )

        print(
            "  rank for 90% energy: {}".format(
                ranks[0.90]
            ),
            flush=True
        )

        print(
            "  shared rank {} cumulative energy: {:.2f}%".format(
                info["shared_rank"],
                100.0
                * info["shared_rank_energy"]
            ),
            flush=True
        )

        print(
            "  rotating rank {} additional energy: {:.2f}%".format(
                info["rotating_rank"],
                100.0
                * info["rotating_rank_energy"]
            ),
            flush=True
        )

        print(
            "  total active rank {} cumulative energy: {:.2f}%".format(
                info["total_rank"],
                100.0
                * info["total_rank_energy"]
            ),
            flush=True
        )

    print(
        "====================================================\n",
        flush=True
    )

class MetaMLPModel(MetaModule):
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits







def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)