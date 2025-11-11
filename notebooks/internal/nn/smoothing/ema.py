"""
Exponential Moving Average (EMA) for model parameters in PyTorch.
"""
from torch import Tensor
from torch.nn import Module

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    EMA (Exponential Moving Average) is a smoothing technique that maintains a moving average of model weights
    during training.

    Formally, at every training step t, the EMA of a parameter theta is updated as:

        theta_{ema, t} = decay * theta_{ema, t-1} + (1 - decay) * theta_t

    Where:
        - theta_t is the current parameter value at step t
        - theta_{ema, t} is the EMA of the parameter at step t. In other words, it's the smoothed version of the parameter.
        - decay is a hyperparameter (0 < decay < 1) that controls the rate of decay.
          A higher decay means the EMA changes more slowly.

    Recent weights have more influence (exponentially decreasing importance for older ones).
    And it acts as a temporal ensemble (similar to averaging multiple checkpoints, but done online during training).

    ---------------

    Note: this technique has been implemented because we have found it in practice to improve model performance during evaluation,
    especially by traders people.

    When we train a neural network, its weights (the "numbers" it learns) change a little bit at every step.
    But these updates can jump around: sometimes the model gets a bit better, sometimes a bit worse
    (and we have seen this happening a lot during training of trading models in this challenge).

    So, instead of using the weights from just the last training step,
    we can take an average of the weights over time to get a more stable version of the model.

    It is like taking an average of multiple snapshots of the model during training.

    For example, if we are tracking one weight (for simplicity) and we have a decay of 0.9:
        - Step t: weight (theta_t)
        - Step 1: weight (theta_1=1.00) ~> theta_{ema, 1} = 1.00
        - Step 2: weight (theta_2=1.20) ~> theta_{ema, 2} = decay * theta{ema, 1} + (1 - decay) * theta_2
                                                          = 0.9 * 1.00 + (1 - 0.9) * 1.20 = 1.02
        - Step 3: weight (theta_3=0.80) ~> theta_{ema, 3} = decay * theta_{ema, 2} + (1 - decay) * theta_3
                                                          = 0.9 * 1.02 + (1 - 0.9) * 0.80 = 0.998
        - Step 4: weight (theta_4=1.50) ~> theta_{ema, 4} = decay * theta_{ema, 3} + (1 - decay) * theta_4
                                                          = 0.9 * 0.998 + (1 - decay) * 1.50 = 1.0482
    And so on... The EMA value moves slowly and smooths out the noisy jumps of the real weight.

    So during evaluation, we can use these smoothed weights (the EMA values) instead of the last weights from training.
    This often leads to better performance because the EMA weights are more stable and
    less affected by the noise of individual training steps.
    """

    def __init__(self, model: Module, decay: float = 0.995):
        """
        Exponential Moving Average (EMA) for model parameters.

        The EMA class maintains a moving average of the model parameters to stabilize training.

        In simple terms, it keeps a smoothed version of the model's weights by averaging them over time,
        which can help improve the model's performance during evaluation.

        However, it does not modify the original model during training; instead, it allows us to apply the smoothed weights
        temporarily during evaluation or inference.
        :param model: The model whose parameters will be tracked.
        :param decay: The decay rate for the moving average. A value close to 1 mean slow updates.
        """
        self.backup: dict[str, Tensor] = {}
        """
        A backup dictionary to store original model parameters when applying EMA weights.
        This allows restoring the original weights after evaluation.
        """

        self.shadow: dict[str, Tensor] = {}
        """
        A dictionary to store the EMA (smoothed) parameters of the model.
        The keys are parameter names, and the values are the corresponding EMA tensors.
        """

        self.decay: float = decay
        """
        The decay rate for the moving average.
        A value close to 1 means slow updates, giving more weight to past parameters.
        """

        # For each parameter in the model, initialize the shadow dictionary with its current value
        for n, p in model.named_parameters():
            # Only track parameters that require gradients, because those are the ones being updated during training
            if p.requires_grad:
                self.shadow[n] = p.detach().clone() # shallow copy of the parameter

    def update(self, model: Module) -> None:
        """
        Update the EMA parameters with the current model parameters.

        This method updates the EMA (Exponential Moving Average) of the model parameters.
        It should be called after each training step to incorporate the latest parameter values into the EMA.
        :param model: The model whose parameters will be used to update the EMA.
        """
        # For each parameter in the model, update the corresponding EMA value
        for n, p in model.named_parameters():
            # Iff the parameter requires gradients (i.e., it's being updated during training)
            if p.requires_grad:
                # Update the EMA value using the formula:
                # shadow[n] = shadow[n] * decay + p * (1 - decay)
                # Where (1 - decay) is the alpha factor for the current parameter
                # Done in-place for efficiency
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)

    def apply_to(self, model: Module) -> None:
        """
        Apply the EMA parameters to the model.

        This method replaces the model's parameters with their EMA (smoothed) versions.
        It is typically used during evaluation or inference to leverage the benefits of the smoothed weights.
        :param model: The model to which the EMA parameters will be applied.
        """
        # Restore original parameters to back-up and replace with EMA parameters
        self.backup = {}
        # For each parameter in the model, replace it with the EMA value
        for n, p in model.named_parameters():
            # Iff the parameter requires gradients (i.e., it's being updated during training)
            if p.requires_grad:
                # Backup the original parameter
                self.backup[n] = p.detach().clone()
                # Replace the parameter with its EMA value (already calculated in update method)
                p.data.copy_(self.shadow[n].data)

    def restore(self, model: Module) -> None:
        """
        Restore the original model parameters.

        This method restores the model's parameters to their original values
        that were backed up before applying the EMA parameters.
        It is used after evaluation or inference to revert the model back to its training state.
        :param model: The model whose parameters will be restored.
        """
        # Restore original parameters from backup
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n].data)
        # Finally, clear the backup dictionary
        self.backup = {}
