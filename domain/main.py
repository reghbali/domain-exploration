# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

import domain.data.data_modules
import domain.model.lit_modules

from domain.cli import DomainCLI


def cli_main():
    cli = DomainCLI()
    # note: don't call fit!!


if __name__ == '__main__':
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
