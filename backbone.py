
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
