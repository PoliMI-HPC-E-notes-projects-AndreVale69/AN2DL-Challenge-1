"""
Implementation of Class-Balanced Focal Loss for multi-class classification tasks.
"""
from numpy import array, bincount, power, maximum, float32 as np_float32, ndarray
from torch import tensor, float32 as torch_float32, softmax, Tensor
from torch.nn import Module, CrossEntropyLoss


class CBFocalLoss(Module):
    """
    Class-Balanced Focal Loss.

    This loss function combines class-balanced weighting with focal loss to address class imbalance
    in multi-class classification tasks. It computes effective number of samples for each class to
    derive class weights, and applies focal loss to focus on hard-to-classify examples.

    Reference:
    Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples.
    https://arxiv.org/abs/1901.05555
    """
    def __init__(self, labels: ndarray, beta: float = 0.999, gamma: float = 2.0, classes: list[int] = (0, 1, 2),
                 alpha: Tensor | None = None
                 ):
        super().__init__()
        self.gamma = gamma
        classes = array(list(classes))
        # counts of each class in the dataset (num_classes,)
        counts  = bincount(labels, minlength=len(classes)).astype(np_float32)
        if alpha is None:
            alpha = tensor([1.0] * len(classes), dtype=torch_float32)
        # apply alpha weights to counts
        counts = counts * alpha.numpy()
        # effective number of samples per class (num_classes,)
        weights = (1.0 - beta) / maximum(1.0 - power(beta, counts), 1e-12)
        # weights normalization (num_classes,)
        weights = weights / weights.sum() * len(classes)
        self.register_buffer('weights', tensor(weights, dtype=torch_float32))
        self.ce = CrossEntropyLoss(reduction='none')

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the Class-Balanced Focal Loss.

        Computes the class-balanced focal loss between the predicted logits and the true target labels.

        1. Computes the cross-entropy loss for each sample.
        2. Computes the predicted probabilities for the true class labels.
        3. Applies the focal loss modulating factor.
        4. Weights the loss for each sample based on class weights.
        5. Returns the mean loss over the batch.
        :param logits: Predicted logits of shape (B, C) where B is batch size and C is number of classes.
        :param target: True class labels of shape (B,).
        :return: Computed class-balanced focal loss as a scalar tensor.
        """
        # Cross-entropy loss per sample (B,)
        ce = self.ce(logits, target)
        # Predicted probability for the true class (B,)
        pt = softmax(logits, dim=1).gather(1, target[:,None]).squeeze(1)
        # Focal loss modulating factor
        focal = (1 - pt).pow(self.gamma) * ce
        # Class weights for each sample (B,)
        w = self.weights[target]
        # Weighted focal loss
        return (w * focal).mean()
