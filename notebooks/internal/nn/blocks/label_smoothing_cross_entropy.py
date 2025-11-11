"""
Label Smoothing Cross-Entropy Loss Module.
"""
from torch import log_softmax, no_grad, zeros_like, Tensor
from torch.nn import Module


class LabelSmoothingCE(Module):
    """
    Label Smoothing Cross-Entropy Loss.

    This loss function implements label smoothing for multi-class classification tasks.
    Instead of using one-hot encoded labels, it smooths the labels by distributing a small
    portion of the probability mass to all classes. This helps to prevent the model from
    becoming overconfident and can improve generalization.
    """
    def __init__(self, smoothing: float=0.05, weight: Tensor | None = None):
        """
        Label Smoothing Cross-Entropy Loss.

        This loss function applies label smoothing to the target labels during training.
        :param smoothing: Smoothing factor to distribute probability mass to non-target classes.
        :param weight: Optional class weights tensor to apply to the loss. Should be of shape (num_classes,).
        """
        super().__init__()
        self.smoothing = smoothing
        # Register weight as a buffer to ensure it is moved to the correct device with the model
        self.register_buffer('weight', weight if weight is not None else None)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the Label Smoothing Cross-Entropy Loss.

        This method computes the loss between the predicted logits and the smoothed target labels.
        :param logits: Predicted logits tensor of shape (batch_size, num_classes).
        :param target: Target labels tensor of shape (batch_size,).
        :return: Computed loss as a scalar tensor.
        """
        # Number of classes in the logits
        n_class = logits.size(-1)
        # Compute log probabilities from logits using log softmax
        log_prob = log_softmax(logits, dim=-1)
        # ~ Create smoothed target distribution
        with no_grad():
            # Initialize true distribution with smoothing
            true_dist = zeros_like(log_prob)
            # Fill non-target classes with smoothing value
            true_dist.fill_(self.smoothing / (n_class - 1))
            # Set target class probability
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        # ~ Compute the loss
        return (
            # Compute loss without class weights
            (-true_dist * log_prob).sum(dim=-1)
            if self.weight is None else
            # Compute loss with class weights (where weight shape is (num_classes,))
            (-true_dist * log_prob * self.weight.unsqueeze(0)).sum(dim=-1)
        )
