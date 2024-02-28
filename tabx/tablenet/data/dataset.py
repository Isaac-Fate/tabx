from typing import Optional
from collections import namedtuple
import shutil
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, random_split

from .transforms import transform_image_to_tensor


MarmotDatasetSample = namedtuple(
    "MarmotDatasetSample",
    (
        "image",
        "table_mask",
        "column_mask",
    ),
)


class MarmotDataset(Dataset):

    def __init__(self, dataset_dir: Path) -> None:

        # Init super class
        super().__init__()

        # Images directory
        images_dir = dataset_dir.joinpath("images")

        # Table masks directory
        table_masks_dir = dataset_dir.joinpath("table-masks")

        # Column masks directory
        column_masks_dir = dataset_dir.joinpath("column-masks")

        # Get all paths of the samples
        self._sample_filepaths = []
        for path in images_dir.glob("*.bmp"):

            # Sample name
            sample_name = path.stem

            # Collect file paths
            self._sample_filepaths.append(
                MarmotDatasetSample(
                    image=images_dir.joinpath(sample_name + ".bmp"),
                    table_mask=table_masks_dir.joinpath(sample_name + ".jpg"),
                    column_mask=column_masks_dir.joinpath(sample_name + ".jpg"),
                ),
            )

    def __len__(self) -> int:

        return len(self._sample_filepaths)

    def __getitem__(self, index: int) -> MarmotDatasetSample:

        # Get the associated file paths of the data sample
        filepaths = self._sample_filepaths[index]

        # Load the image

        # Read BMP iamge
        image_filepath: Path = filepaths.image
        image = Image.open(image_filepath)

        # Convert to tensor
        image_tensor = transform_image_to_tensor(image)

        # Load table mask
        table_mask_filepath: Path = filepaths.table_mask
        table_mask = Image.open(table_mask_filepath)

        # Convert to tensor
        table_mask_tensor = transform_image_to_tensor(table_mask)

        # Load column mask
        column_mask_filepath: Path = filepaths.column_mask
        column_mask = Image.open(column_mask_filepath)

        # Convert to tensor
        column_mask_tensor = transform_image_to_tensor(column_mask)

        return MarmotDatasetSample(
            image=image_tensor,
            table_mask=table_mask_tensor,
            column_mask=column_mask_tensor,
        )
