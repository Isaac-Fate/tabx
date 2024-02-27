from typing import Self, Optional
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import wandb
from loguru import logger

from .config import TrainingConfig
from ..logging import TrainingLogRecord, ValidationLogRecord
from ..data import MarmotDataset
from ..models import TableNet, Mask
from ..criterion import dice_loss


class Trainer:

    def __init__(self, config: TrainingConfig) -> None:

        self._config = config

        # Set up logger
        self._logger = logger.bind(key="table-net")
        self._logger.add(self._config.log_filepath)

        # Check if CUDA (GPU) is available, else use CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self._dataset: Optional[MarmotDataset] = None
        self._train_dataloader: Optional[DataLoader] = None
        self._valid_dataloader: Optional[DataLoader] = None
        self._test_dataloader: Optional[DataLoader] = None

        # Create table net model
        self._model = TableNet()

        # Move the model to device
        self._model.to(self._device)

    @property
    def device(self) -> str:

        return self._device

    @property
    def model(self) -> TableNet:

        return self._model

    @classmethod
    def from_config_file(cls, filepath: Path | str) -> Self:

        # Get the run dir
        run_dir = filepath.parent

        # Read the configuration file
        with open(filepath, "rb") as fp:
            data = tomllib.load(fp)

        # Set run dir
        data["run_dir"] = run_dir

        # Convert to TrainingConfig
        config = TrainingConfig.model_validate(data)

        return cls(config)

    def train(self):

        # Log in to Wandb
        wandb.login()

        # Create a run
        run = wandb.init(
            # Project name
            project="table-net",
            # Run name
            name=self._config.run_name,
        )

        # Prepare data
        self._prepare_data()

        # Optimizer
        optimizer = Adam(
            self._model.parameters(),
            lr=self._config.adam_lr,
        )

        # Log
        self._logger.info("Start training...")
        self._logger.info(f"Training Configuration: {self._config}")

        # Training loop
        for i in range(self._config.n_epochs):

            # Epoch number
            epoch = i + 1

            n_batches = len(self._train_dataloader)
            for i, item in enumerate(self._train_dataloader):

                # Batch number
                batch = i + 1

                # Zero the gradients
                optimizer.zero_grad()

                # Unpack item, and
                # move data to device
                image = item.image.to(self._device)
                table_mask = item.table_mask.to(self._device)
                column_mask = item.column_mask.to(self._device)

                # Forward pass
                # Get both prdicted table and column masks
                mask: Mask = self._model(image)

                # Compute dice losses
                table_loss = dice_loss(mask.table, table_mask)
                column_loss = dice_loss(mask.column, column_mask)

                # Overall loss
                loss = (table_loss + column_loss) / 2

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                # Log

                # Log record
                train_log_record = TrainingLogRecord(
                    epoch=epoch,
                    batch=batch,
                    n_batches=n_batches,
                    train_loss=loss.item(),
                )

                # Loguru
                self._logger.info(train_log_record.to_message())

                # Wandb
                wandb.log(train_log_record.model_dump())

            # Validate
            self._validate(epoch)

            # Save checkpoint
            self._save_checkpoint(epoch)

    def _prepare_data(self) -> None:

        # Load dataset
        dataset = MarmotDataset(self._config.dataset_dir)

        # Split the dataset
        train_dataset, valid_dataset, test_dataset = dataset.split(
            train=self._config.train_ratio,
            valid=self._config.valid_ratio,
            test=self._config.test_ratio,
        )

        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=self._config.batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self._config.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self._config.batch_size)

        # Log
        self._logger.info("Done preparing data")
        self._logger.info(f"Training item names: {dataset.train_item_names}")
        self._logger.info(f"Validation item names: {dataset.valid_item_names}")
        self._logger.info(f"Test item names: {dataset.test_item_names}")

        self._dataset = dataset
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    def _validate(self, epoch: int) -> None:
        with torch.no_grad():
            # List of loss of each batch
            valid_losses = []

            # Validation loop
            n_batches = len(self._valid_dataloader)
            for item in self._valid_dataloader:
                # Unpack item
                image = item.image.to(self._device)
                table_mask = item.table_mask.to(self._device)
                column_mask = item.column_mask.to(self._device)

                # Get both prdicted table and column masks
                mask: Mask = self._model(image)

                # Compute the dice coefficient of both table and column
                table_loss = dice_loss(mask.table, table_mask)
                column_loss = dice_loss(mask.column, column_mask)

                # Validation performance of this batch
                valid_dice_loss = (table_loss + column_loss) / 2

                # Add to list
                valid_losses.append(valid_dice_loss.item())

            # Overall validation performance
            valid_loss = sum(valid_losses) / n_batches

            # Log

            # Log record
            valid_log_record = ValidationLogRecord(
                epoch=epoch,
                valid_loss=valid_loss,
            )

            # Loguru
            logger.info(valid_log_record.to_message())

            # Wandb
            wandb.log(valid_log_record.model_dump())

    def _save_checkpoint(self, epoch: int) -> None:

        # No need to save
        if (
            epoch % self._config.save_every_n_epochs != 0
            and epoch != self._config.n_epochs
        ):
            return

        # State dict
        state_dict = self._model.state_dict()

        # File path
        checkpoint_filepath = self._config.checkpoints_dir.joinpath(
            f"checkpoint-{epoch}"
        ).with_suffix(".pt")

        # Save
        torch.save(state_dict, checkpoint_filepath)

        # Log
        self._logger.info(f"Checkpoint is saved at: {checkpoint_filepath}")
