"""
Focal Loss implementation in PyTorch.
"""
from torch import log_softmax, exp, Tensor
from torch.nn import Module
from enum import Enum


class ReductionMode(Enum):
    """ Reduction modes for loss computation. """

    MEAN = 'mean'
    """ Return the mean of the loss values. """

    SUM = 'sum'
    """ Return the sum of the loss values. """


class FocalLoss(Module):
    """
    Focal Loss for multi-class classification.

    This loss function is designed to address class imbalance by down-weighting easy examples
    and focusing more on hard examples. It introduces a modulating factor (1 - p_t)^gamma to the
    standard cross-entropy loss, where p_t is the model's estimated probability for the true class.
    An optional alpha parameter can be used to weight the loss for each class.
    """

    def __init__(
            self,
            alpha: Tensor | None = None,
            gamma: float = 2.0,
            reduction: ReductionMode | None = ReductionMode.MEAN
    ):
        """
        Focal Loss for multi-class classification.

        This loss function is designed to address class imbalance by down-weighting easy examples
        and focusing more on hard examples. It introduces a modulating factor (1 - p_t)^gamma to the
        standard cross-entropy loss, where p_t is the model's estimated probability for the true class.
        An optional alpha parameter can be used to weight the loss for each class.
        :param alpha: Class weights tensor of shape (C,) where C is the number of classes. If None, no class weighting is applied.
        :param gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        :param reduction: Reduction mode to apply to the output: 'mean', 'sum', or None for no reduction.
        """
        super().__init__()
        self.gamma: float = gamma
        self.reduction: ReductionMode | None = reduction
        # Register alpha as a buffer to ensure it's moved to the correct device with the model
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the Focal Loss.

        Computes the focal loss between the predicted logits and the true target labels.

        The steps involved are:
         1. Computes the log-probabilities and probabilities of the predicted classes.
         2. Gathers the log-probabilities and probabilities corresponding to the true class labels.
         3. Applies the focal loss formula: - (1 - p_t)^gamma * log(p_t).
         4. If alpha is provided, scales the loss by the class weights.
         5. Applies the specified reduction method to the loss values.

        :param logits: Predicted logits tensor of shape (B, C) where B is the batch size and C is the number of classes.
        :param target: True class labels tensor of shape (B,) with values in the range [0, C-1].
        :return: Computed focal loss as a scalar tensor if reduction is applied, otherwise a tensor of shape (B,).
        """
        # Get log-probabilities over classes (B,C)
        log_probs = log_softmax(logits, dim=-1)
        # Get probabilities over classes (B,C)
        probs = exp(log_probs)
        # Gather log-probabilities and probabilities for the true class labels (B,)
        log_probs_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        # Gathered probabilities for the true class labels (B,)
        pt_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        # Compute the focal loss
        loss = - (1 - pt_t) ** self.gamma * log_probs_t
        # Apply class weights if provided
        if self.alpha is not None:
            # Scale the loss by the class weights
            loss *= self.alpha[target]
        # Apply reduction method
        return (
            loss.mean() if self.reduction == ReductionMode.MEAN else
            loss.sum() if self.reduction == ReductionMode.SUM else
            loss
        )
