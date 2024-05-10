import lightning as L


class DomainCLI(L.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            'data.num_classes', 'model.num_classes', apply_on='instantiate'
        )
        parser.link_arguments(
            'data.image_shape', 'model.input_shape', apply_on='instantiate'
        )
