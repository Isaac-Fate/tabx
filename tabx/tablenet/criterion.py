import torch
from torch import Tensor


def dice_coeff(
    pred: Tensor,
    target: Tensor,
    smooth: float = 1.0,
) -> Tensor:
    """Dice coefficient.
    The higher the value is,
    the more accurate the prediction is.

    Parameters
    ----------
    pred : Tensor
        Predicted mask.
    target : Tensor
        Grand truth mask.
    smooth : float, optional
        Smoothing factor, by default 1.0.

    Returns
    -------
    Tensor
        Dice coefficient.
    """

    # Flatten the tensors
    pred = pred.flatten()
    target = target.flatten()

    # Compute intersection
    # It is given by the inner product of the predicted mask and
    # the ground truth
    intersection = torch.dot(pred, target)

    # Compute dice value
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def dice_loss(
    pred: Tensor,
    target: Tensor,
    smooth: float = 1.0,
) -> Tensor:
    """The Dice loss. It is given by 1 - Dice coefficient.

    Parameters
    ----------
    pred : Tensor
        Predicted mask.
    target : Tensor
        Grand truth mask.
    smooth : float, optional
        Smoothing factor, by default 1.0.

    Returns
    -------
    Tensor
        Dice loss.
    """

    # Compute the dice coefficient
    dice = dice_coeff(
        pred,
        target,
        smooth,
    )

    return 1 - dice
