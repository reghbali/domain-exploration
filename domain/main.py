# main.py
from lightning.pytorch.cli import LightningCLI

import domain.data.data_modules
import domain.model.lit_modules


def cli_main():
    cli = LightningCLI()
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
