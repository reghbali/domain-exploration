from lightning.pytorch.cli import LightningCLI


class DomainCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            'data.num_classes', 'model.init_args.num_classes', apply_on='instantiate'
        )
        parser.link_arguments(
            'data.image_shape',
            'model.init_args.net.input_shape',
            apply_on='instantiate',
        )
