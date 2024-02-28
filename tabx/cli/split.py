from typing import Annotated
from pathlib import Path
import typer
from .app import app


@app.command(help="Split the dataset.")
def split(
    marmot_dataset_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Directory of the Marmot dataset.",
        ),
    ],
    data_subsets_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Root directory of the data subsets.",
        ),
    ],
    train_ratio: Annotated[
        float,
        typer.Option("--train", help="Proportion of the training dataset."),
    ] = 0.9,
    valid_ratio: Annotated[
        float,
        typer.Option("--valid", help="Proportion of the validation dataset."),
    ] = 0.05,
    test_ratio: Annotated[
        float,
        typer.Option("--test", help="Proportion of the test dataset."),
    ] = 0.05,
):
    from tabx.tablenet.data import MarmotDatasetSplitter

    # Create a splitter
    splitter = MarmotDatasetSplitter(marmot_dataset_dir)

    # Split the dataset
    splitter.split(
        train=train_ratio,
        valid=valid_ratio,
        test=test_ratio,
        data_subsets_dir=data_subsets_dir,
    )
