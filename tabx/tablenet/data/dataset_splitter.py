from typing import Optional
from collections import namedtuple
import shutil
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, random_split

from .transforms import transform_image_to_tensor


MarmotDatasetItem = namedtuple(
    "MarmotDatasetItem",
    (
        "image",
        "table_mask",
        "column_mask",
    ),
)

SplitResult = namedtuple(
    "SplitResult",
    (
        "train",
        "valid",
        "test",
    ),
)


class MarmotDatasetSplitter(Dataset):

    def __init__(self, marmot_dataset_dir: Path | str) -> None:

        # Init super class
        super().__init__()

        # Convert to path
        marmot_dataset_dir = Path(marmot_dataset_dir)
        self._dataset_dir = marmot_dataset_dir

        # Table masks dir
        table_masks_dir = marmot_dataset_dir.joinpath("table-masks")

        # Column masks dir
        column_masks_dir = marmot_dataset_dir.joinpath("column-masks")

        # File paths of all data items
        self._item_filepaths: list[MarmotDatasetItem] = []

        # A dict that maps the index of the name of each data item
        self._index_to_sample_name: dict[int, str] = {}

        # Indices of the training, validation, test datasets
        self._train_indices: Optional[tuple[int]] = None
        self._valid_indices: Optional[tuple[int]] = None
        self._test_indices: Optional[tuple[int]] = None

        for index, xml_filepath in enumerate(marmot_dataset_dir.glob("*.xml")):

            # Item name
            sample_name = xml_filepath.stem

            # Add to dict
            self._index_to_sample_name[index] = sample_name

            """To be fair, The naming of Marmot dataset is pretty BAD!
            For example, 10.1.1.1.2006_3.bmp includes the dot "." in its path,
            which will lead to unexpected results when using `Path` to process the paths.
            """

            # Image file path
            image_filepath = Path(
                str(marmot_dataset_dir.joinpath(sample_name)) + ".bmp"
            )

            # Associated table mask image file path
            table_mask_filepath = Path(
                str(table_masks_dir.joinpath(sample_name)) + ".jpg"
            )

            # Associated column mask image file path
            column_mask_filepath = Path(
                str(column_masks_dir.joinpath(sample_name)) + ".jpg"
            )

            # Add to paths
            self._item_filepaths.append(
                MarmotDatasetItem(
                    image=image_filepath,
                    table_mask=table_mask_filepath,
                    column_mask=column_mask_filepath,
                )
            )

    def __len__(self) -> int:

        return len(self._item_filepaths)

    def __getitem__(self, index: int) -> MarmotDatasetItem:

        # Get the associated file paths of the data item
        filepaths = self._item_filepaths[index]

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

        return MarmotDatasetItem(
            image=image_tensor,
            table_mask=table_mask_tensor,
            column_mask=column_mask_tensor,
        )

    @property
    def train_indices(self) -> Optional[tuple]:
        """Indices of items in training dataset."""

        return self._train_indices

    @property
    def valid_indices(self) -> Optional[tuple]:
        """Indices of items in validation dataset."""

        return self._valid_indices

    @property
    def test_indices(self) -> Optional[tuple]:
        """Indices of items in test dataset."""

        return self._test_indices

    @property
    def train_sample_names(self) -> Optional[tuple]:
        """Names of items in training dataset."""

        if self._train_indices is None:
            return None

        return tuple(
            map(
                lambda index: self._index_to_sample_name[index],
                self._train_indices,
            )
        )

    @property
    def valid_sample_names(self) -> Optional[tuple]:
        """Names of items in validation dataset."""

        if self._valid_indices is None:
            return None

        return tuple(
            map(
                lambda index: self._index_to_sample_name[index],
                self._valid_indices,
            )
        )

    @property
    def test_sample_names(self) -> Optional[tuple]:
        """Names of items in test dataset."""

        if self._test_indices is None:
            return None

        return tuple(
            map(
                lambda index: self._index_to_sample_name[index],
                self._test_indices,
            )
        )

    def split(
        self,
        *,
        train: float,
        valid: float = 0.0,
        test: float = 0.0,
        data_subsets_dir: Optional[Path] = None
    ) -> SplitResult:
        """Split the dataset into training, validation and test datasets.
        The proportion of training data set must be positive.
        The returned validation or test dataset may be `None`
        if the associated proportion is zero.

        Parameters
        ----------
        train : float
            Proportion of the training dataset.
        valid : float, optional
            Proportion of the validation dataset, by default 0.0
        test : float, optional
            Proportion of the test dataset, by default 0.0
        data_subsets_dir : Optional[Path], optional
            If this is set, then data subsets will be created under
            the provided directory.

        Returns
        -------
        SplitResult
            - train: Training dataset.
            - valid: Valid dataset.
            - test: Test dataset.
        """

        # Check input

        assert (
            train > 0 and valid >= 0 and test >= 0
        ), "Proportion of each subset must be nonnegative and the proporting of the training dataset must be positive"

        assert train + valid + test == 1.0, "The sum of the proportions must equal 1"

        # Split dataset

        if valid == 0 and test == 0:
            (train_dataset,) = random_split(self, (train,))
            valid_dataset = None
            test_dataset = None

            # Set indices
            self._train_indices = tuple(train_dataset.indices)

        elif valid == 0 and test > 0:
            train_dataset, test_dataset = random_split(self, (train, test))
            valid_dataset = None

            # Set indices
            self._train_indices = tuple(train_dataset.indices)
            self._test_indices = tuple(test_dataset.indices)

        elif valid > 0 and test == 0:
            train_dataset, valid_dataset = random_split(self, (train, valid))
            test_dataset = None

            # Set indices
            self._train_indices = tuple(train_dataset.indices)
            self._valid_indices = tuple(valid_dataset.indices)

        else:
            # All three subsets exist
            train_dataset, valid_dataset, test_dataset = random_split(
                self, (train, valid, test)
            )

            # Set indices
            self._train_indices = tuple(train_dataset.indices)
            self._valid_indices = tuple(valid_dataset.indices)
            self._test_indices = tuple(test_dataset.indices)

        # Create data subsets
        if data_subsets_dir is not None:
            self.create_data_subsets(data_subsets_dir)

        return SplitResult(
            train=train_dataset,
            valid=valid_dataset,
            test=test_dataset,
        )

    def create_data_subsets(
        self,
        data_subsets_dir: Path,
    ) -> None:

        if not data_subsets_dir.is_dir():
            data_subsets_dir.mkdir()

        if self.train_sample_names is not None:
            train_dataset_dir = data_subsets_dir.joinpath("train")
            if not train_dataset_dir.is_dir():
                train_dataset_dir.mkdir()
            self._create_data_subset(
                train_dataset_dir,
                sample_names=self.train_sample_names,
            )

        if self.valid_sample_names is not None:
            valid_dataset_dir = data_subsets_dir.joinpath("valid")
            if not valid_dataset_dir.is_dir():
                valid_dataset_dir.mkdir()
            self._create_data_subset(
                valid_dataset_dir,
                sample_names=self.valid_sample_names,
            )

        if self.test_sample_names is not None:
            test_dataset_dir = data_subsets_dir.joinpath("test")
            if not test_dataset_dir.is_dir():
                test_dataset_dir.mkdir()
            self._create_data_subset(
                test_dataset_dir,
                sample_names=self.test_sample_names,
            )

    def _create_data_subset(
        self,
        data_subset_dir: Path,
        sample_names: tuple[str],
    ) -> None:

        # Create the dir if it does not exist
        if not data_subset_dir.is_dir():
            data_subset_dir.mkdir()

        # Images dir
        images_dir = data_subset_dir.joinpath("images")
        if not images_dir.is_dir():
            images_dir.mkdir()

        # Copy images
        for sample_name in sample_names:
            shutil.copy(
                self._dataset_dir.joinpath(sample_name + ".bmp"),
                images_dir.joinpath(sample_name + ".bmp"),
            )

        # Table masks dir
        table_masks_dir = data_subset_dir.joinpath("table-masks")
        if not table_masks_dir.is_dir():
            table_masks_dir.mkdir()

        # Copy table masks
        for sample_name in sample_names:
            shutil.copy(
                self._dataset_dir.joinpath("table-masks").joinpath(
                    sample_name + ".jpg"
                ),
                table_masks_dir.joinpath(sample_name + ".jpg"),
            )

        # Column masks dir
        column_masks_dir = data_subset_dir.joinpath("column-masks")
        if not column_masks_dir.is_dir():
            column_masks_dir.mkdir()

        # Copy column masks
        for sample_name in sample_names:
            shutil.copy(
                self._dataset_dir.joinpath("column-masks").joinpath(
                    sample_name + ".jpg"
                ),
                column_masks_dir.joinpath(sample_name + ".jpg"),
            )
