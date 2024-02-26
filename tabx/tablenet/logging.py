from pydantic import BaseModel


class TrainingLogRecord(BaseModel):

    epoch: int
    batch: int
    n_batches: int
    train_loss: float

    def to_message(self) -> str:
        """Convert to a message to log.

        Returns
        -------
        str
            Message of this record.
        """

        return (
            "Training - "
            "Epoch: {epoch} - "
            "Batch: {batch}/{n_batches} - "
            "Training Loss: {train_loss}"
        ).format(
            epoch=self.epoch,
            batch=self.batch,
            n_batches=self.n_batches,
            train_loss=self.train_loss,
        )


class ValidationLogRecord(BaseModel):

    epoch: int
    valid_loss: float

    def to_message(self) -> str:
        """Convert to a message to log.

        Returns
        -------
        str
            Message of this record.
        """

        return ("Validation - Epoch: {epoch} - Validation Loss: {valid_loss}").format(
            epoch=self.epoch,
            valid_loss=self.valid_loss,
        )
