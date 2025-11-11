"""
Implementation of Class-Balanced Focal Loss for multi-class classification tasks.
"""
from numpy import bincount, float32 as np_float32, ndarray
from torch import tensor, float32 as torch_float32, softmax, Tensor, ones, as_tensor, pow as torch_pow, clamp
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
                 alpha: Tensor | ndarray | list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        num_classes = len(classes)

        # counts per class as torch tensor (CPU is fine; we'll register buffer later)
        counts = bincount(labels, minlength=num_classes).astype(np_float32)
        counts = tensor(counts, dtype=torch_float32)

        # alpha -> torch.float32 tensor
        if alpha is None:
            alpha_t = ones(num_classes, dtype=torch_float32)
        else:
            alpha_t = as_tensor(alpha.cpu(), dtype=torch_float32)

        # apply alpha to counts BEFORE effective-number weighting
        counts *= alpha_t

        # effective number weights: (1 - beta) / (1 - beta^n_c)
        beta_t = tensor(beta, dtype=torch_float32)
        # guard against zero counts
        denominator = 1.0 - torch_pow(beta_t, clamp(counts, min=1e-12))
        weights = (1.0 - beta_t) / clamp(denominator, min=1e-12)

        # normalize to mean 1.0 (optional but common)
        weights = weights / weights.sum() * num_classes

        # register as buffer so it moves with .to(device)
        self.register_buffer("weights", weights)


        # per-sample CE
        self.ce = CrossEntropyLoss(reduction="none")

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
        # Ensure correct dtypes
        target = target.long()

        # Per-sample CE (B,)
        ce = self.ce(logits, target)

        # p_t for the true class (B,)
        pt = softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        # numerical safety
        pt = pt.clamp_(1e-6, 1 - 1e-6)

        # Focal modulation
        focal = (1.0 - pt).pow(self.gamma) * ce

        # Class weights per sample; make sure buffer is on same device as logits
        w = self.weights.to(logits.device).index_select(0, target)

        return (w * focal).mean()
