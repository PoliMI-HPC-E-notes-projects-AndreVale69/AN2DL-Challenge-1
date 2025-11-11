"""
Trainer for the PainTCNBiLSTMAttn model.
"""
import json
from itertools import product

from numpy import array, float32 as numpy_float32, concatenate, round as np_round, linspace, zeros
from numpy import ndarray
from numpy.random import beta
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch import device as torch_device, tensor, no_grad, Tensor, randperm, cat, save, float32 as torch_float32
from torch.cuda import is_available as cuda_is_available
from torch.nn import Module, CrossEntropyLoss
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from internal.nn.blocks.class_balanced_focal_loss import CBFocalLoss
from internal.nn.models.pain_tcn_bilstm_attn import PainTCNBiLSTMAttn
from internal.nn.smoothing.ema import EMA


class PainTCNBiLSTMAttnTrainer:
    """
    Trainer for the PainTCNBiLSTMAttn model.

    This class encapsulates the training logic for the PainTCNBiLSTMAttn model, including
    warm-up phases, main training loops, evaluation metrics, temperature scaling, and logit bias
    search. It supports mixup data augmentation and uses an Exponential Moving Average (EMA)
    of model parameters for improved evaluation stability.
    """
    def __init__(
            self,
            classes: list[int],
            epochs: int = 160,
            patience: int = 24,
            warmup_epochs: int = 5,
            p_drop_summ: float = 0.3,
            summary_window: int = 9,
            device: torch_device = torch_device('cuda' if cuda_is_available() else 'cpu')
    ):
        """
        Trainer for the PainTCNBiLSTMAttn model.
        :param classes: List of class labels.
        :param epochs: Number of training epochs.
        :param patience: Early stopping patience.
        :param warmup_epochs: Number of warm-up epochs.
        :param p_drop_summ: Probability of dropping summary input during training.
        :param summary_window: Size of the summary window.
        :param device: Device to run the training on. If None, uses CUDA if available.
        """
        self._classes = classes
        self._epochs = epochs
        self._patience = patience
        self._warmup_epochs = warmup_epochs
        self._p_drop_summ = p_drop_summ
        self._summary_window = summary_window
        self._device = device

    def train_one_fold(
            self,
            model: PainTCNBiLSTMAttn,
            k: int,
            train_loader: DataLoader,
            val_loader: DataLoader,
            y_train: ndarray,
            y_val: ndarray,
            class_multipliers: dict[int, float]
    ) -> tuple[PainTCNBiLSTMAttn, float, ndarray, float]:
        """
        Train the model for one fold of cross-validation.

        This method handles the training process for a single fold, including warm-up,
        main training loop, evaluation, temperature scaling, and logit bias search.
        :param model: The PainTCNBiLSTMAttn model to be trained.
        :param k: Fold index.
        :param train_loader: Training data loader.
        :param val_loader: Validation data loader.
        :param y_train: The training labels.
        :param y_val: The validation labels.
        :param class_multipliers: Dictionary of class multipliers for class weighting.
        :return: Tuple containing the trained model, final macro F1 score, best logit bias, and learned temperature.
        Where the best logit bias indicates the optimal bias added to logits for each class to maximize macro F1 score on validation data.
        And the learned temperature is a scalar used for temperature scaling of logits.
        """
        # ----- per-fold class weights (with class-2 boost) -----
        base_w = compute_class_weight('balanced', classes=array(self._classes.copy()), y=y_train).astype(numpy_float32)
        base_w[2] *= class_multipliers.get(2, 1.0)  # boost class 2 if specified

        # --- WARM-UP: freeze summary branch and/or hard-mask it ---
        self._set_requires_grad(model.summ_mlp, False)
        model.p_drop_summ = 1.0  # always drop summary during warm-up

        # build optimizer on current trainable params
        criterion = CBFocalLoss(y_train, beta=0.999, gamma=2.0, alpha=tensor(base_w, dtype=torch_float32, device=self._device))
        optimizer = AdamW(self._build_param_groups(model), lr=1e-3)
        scheduler = ReduceLROnPlateau(
            optimizer, # optimizer
            mode='max',
            factor=0.5,
            patience=5,
            cooldown=3,  # after lr reduction, wait this many epochs before resuming normal operation
            min_lr=2e-5
        )

        # ---- warm-up loop ----
        best_f1, best_state, bad = -1.0, None, 0
        for epoch in range(1, self._warmup_epochs + 1):
            model.train()
            running = 0.0
            for batch in train_loader:
                x_num, x_sta, x_surv, x_summ, y = self._get_data_from_batch(batch)
                optimizer.zero_grad()
                logits = model(x_num, x_surv, x_sta, x_summ)
                loss = criterion(logits, y)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running += loss.item() * y.size(0)

            train_loss = running / len(train_loader.dataset)

            # --- validation ---
            macro_f1, rec_pc, cm = self._evaluate_metrics(model, val_loader)
            scheduler.step(macro_f1)

            # where lr means current learning rate
            # and rec is per-class recall
            print(f"[F{k} {epoch:03d}] t_loss={train_loss:.4f} | F1(macro)={macro_f1:.4f} | "
                  f"rec={np_round(rec_pc, 3)} | lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"patience={bad + 1}/{self._patience}")

            # early stopping on macro-F1
            if macro_f1 > best_f1:
                best_f1, bad = macro_f1, 0
            else:
                bad += 1
                if bad >= self._patience:
                    print(f"[F{k}] Early stopping.")
                    break

        # --- UNFREEZE summaries & set chosen drop prob (e.g., 0.3 or 0.7) ---
        self._set_requires_grad(model.summ_mlp, True)
        model.p_drop_summ = self._p_drop_summ

        # re-build optimizer so the newly-unfrozen params get included with correct WD
        param_groups = self._build_param_groups(model)
        optimizer = AdamW(param_groups, lr=1e-3)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            cooldown=3,
            min_lr=2e-5
        )

        # EMA (Exponential Moving Average) of parameters
        ema = EMA(model, decay=0.999)

        # ---- main training loop ----
        best_f1, best_state, bad = -1.0, None, 0
        for epoch in range(1, self._epochs + 1):
            model.train()
            running = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                # mixup augmentation
                batch = {
                    key: (val.to(self._device) if not isinstance(val, list) else [t.to(self._device) for t in val])
                    for key, val in batch.items()
                }
                mixed, ya, yb, lam = self._mixup_batch(batch, alpha=0.2)

                logits = model(mixed['x_num'], mixed['x_surv'], mixed['x_sta'], mixed['x_summ'])
                ce = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)  # cross-entropy loss with mixup

                # --- L1 regularization on summary MLP weights ---
                l1 = 0.0
                # first Linear in summ_mlp;
                if any(p.requires_grad for p in model.summ_mlp.parameters()):
                    l1 = 1e-6 * model.summ_mlp[1].weight.abs().sum()
                loss = ce + l1

                # --- backprop ---
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # update EMA
                ema.update(model)
                running += loss.item() * y.size(0)

            train_loss = running / len(train_loader.dataset)

            # --- validation with EMA weights ---
            ema.apply_to(model)
            macro_f1, rec_pc, cm = self._evaluate_metrics(model, val_loader)
            ema.restore(model)  # restore original weights
            scheduler.step(macro_f1)

            # where lr means current learning rate
            # and rec is per-class recall
            print(f"[F{k} {epoch:03d}] t_loss={train_loss:.4f} | F1(macro)={macro_f1:.4f} | "
                  f"rec={np_round(rec_pc, 3)} | lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"patience={bad + 1}/{self._patience}")

            # early stopping on macro-F1
            if macro_f1 > best_f1:
                best_f1, bad = macro_f1, 0
                best_state = {kk: vv.cpu() for kk, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self._patience:
                    print(f"[F{k}] Early stopping.")
                    break

        # restore best
        model.load_state_dict({kk: vv.to(self._device) for kk, vv in best_state.items()})

        t_fold = self._fit_temperature_on_val(model, val_loader, ema)
        print(f"[F{k}] Learned temperature T = {t_fold:.3f}")
        best_b, f1_biased = self._search_logit_bias(model, val_loader, y_val)
        print("Best logit bias:", best_b, "biased macro-F1:", f1_biased)
        final_f1, final_rec, final_cm = self._evaluate_metrics(model, val_loader)
        save(best_state, f'artifacts/fold{k}_best_K_{class_multipliers.get(2, 1)}mul.pt')
        try:
            json.dump({
                'macroF': final_f1,
                'recall': final_rec.tolist(),
                'cm': final_cm.tolist(),
                'window': self._summary_window
            }, open(f'artifacts/fold{k}_metrics_{class_multipliers.get(2, 1)}mul.json', 'w'), indent=4)
        except Exception as e:
            print("Could not save metrics json:", e)
        print(f"[F{k}] BEST macroF={final_f1:.4f}  saved=artifacts/fold{k}_best_K_{class_multipliers.get(2, 1)}mul.pt")
        return model, final_f1, best_b, t_fold

    def _get_data_from_batch(self, batch: dict) -> tuple[Tensor, Tensor, list[Tensor], Tensor, Tensor]:
        """
        Extract and move batch data to the appropriate device.
        :param batch: Batch dictionary containing input tensors.
        :return: Tuple of tensors moved to the device.
        """
        return (
            batch['x_num'].to(self._device),
            batch['x_sta'].to(self._device),
            [s.to(self._device) for s in batch['x_surv']],
            batch['x_summ'].to(self._device),
            batch['y'].to(self._device)
        )

    @staticmethod
    def _set_requires_grad(module: Module, flag: bool) -> None:
        """
        Set requires_grad for all parameters in a module.
        :param module: The module whose parameters' requires_grad will be set.
        :param flag: Boolean flag to set requires_grad to.
        """
        for p in module.parameters():
            p.requires_grad = flag

    @staticmethod
    def _build_param_groups(model: PainTCNBiLSTMAttn) -> list[dict[str, list[Parameter] | float]]:
        """
        Build parameter groups with different weight decay settings.
        1. Summary MLP weights: weight decay = 5e-4
        2. Other weights: weight decay = 1e-4
        3. No decay (biases, LayerNorm/BatchNorm weights): weight decay = 0.0
        :param model: The model whose parameters will be grouped.
        :return: List of parameter groups for the optimizer. Includes three groups with specified weight decay.
        """
        decay_summ: list[Parameter] = []
        decay_other: list[Parameter] = []
        no_decay: list[Parameter] = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Biases & (LayerNorm/BatchNorm/etc.) weights are 1-D -> no weight decay
            is_no_decay = (p.dim() == 1) or n.endswith('.bias') or 'norm' in n.lower() or 'bn' in n.lower()

            if is_no_decay:
                no_decay.append(p)
            elif n.startswith('summ_mlp'):
                decay_summ.append(p)
            else:
                decay_other.append(p)

        # Optional sanity checks to catch overlaps or misses
        ids: list[int] = [id(x) for x in decay_summ + decay_other + no_decay]
        assert len(ids) == len(set(ids)), "A parameter landed in more than one group."
        total: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
        covered: int = sum(p.numel() for p in decay_summ + decay_other + no_decay)
        assert total == covered, f"Some params not assigned: total {total} vs covered {covered}"

        return [
            {'params': decay_summ, 'weight_decay': 5e-4},
            {'params': decay_other, 'weight_decay': 1e-4},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    @staticmethod
    def _mixup_batch(batch: dict, alpha: float = 0.2) -> tuple[dict[str, Tensor | list[Tensor]], Tensor, Tensor, float]:
        """
        Apply mixup augmentation to a batch.

        The mixup technique creates new training samples by combining pairs of examples and their labels.
        This is done by sampling a mixing coefficient from a Beta distribution and linearly combining
        the inputs and labels of two randomly selected examples in the batch.

        It helps improve model generalization and robustness by encouraging the model to behave linearly
        between training examples.

        For example, given two samples (x1, y1) and (x2, y2), the mixed sample is:
            x_mixed = lam * x1 + (1 - lam) * x2
            y_mixed = lam * y1 + (1 - lam) * y2
        :param batch: Batch dictionary containing input tensors and labels.
        :param alpha: Parameter for the Beta distribution to sample the mixing coefficient.
        :return: Tuple containing the mixed batch, original labels, permuted labels, and mixing coefficient.
        1. mixed batch: Dictionary with mixed input tensors.
        2. original labels: Tensor of original labels before mixing.
        3. permuted labels: Tensor of labels corresponding to the mixed inputs.
        4. mixing coefficient: Float value used for mixing the inputs and labels.
        """
        lam: float = beta(alpha, alpha) if alpha > 0 else 1.0
        idx: Tensor = randperm(batch['x_num'].size(0), device=batch['x_num'].device)

        def mix(a: Tensor) -> Tensor:
            return lam * a + (1 - lam) * a[idx]

        mixed: dict[str, Tensor | list[Tensor]] = {
            'x_num': mix(batch['x_num']),
            'x_sta': mix(batch['x_sta']),
            'x_summ': mix(batch['x_summ']),
            'x_surv': [mix(s.float()).long() for s in batch['x_surv']]
            # if embeddings expect long, skip mix on surveys
        }
        y_a, y_b = batch['y'], batch['y'][idx]
        return mixed, y_a, y_b, lam

    def _fit_temperature_on_val(self, model: PainTCNBiLSTMAttn, val_loader: DataLoader, ema: EMA | None = None, init_t: float = 1.0):
        """
        Fit temperature scaling on validation data.

        Temperature scaling is a post-processing technique used to calibrate the predicted probabilities
        of a classification model. It involves introducing a temperature parameter T that scales the logits
        before applying the softmax function. A higher temperature (>1) produces a softer probability distribution,
        while a lower temperature (<1) makes the distribution peakier.

        This method optimizes the temperature parameter T on the validation set to minimize the
        negative log-likelihood (NLL) loss, thereby improving the calibration of the model's predictions.

        It is particularly useful when the model is overconfident in its predictions, as it helps
        adjust the confidence levels to better reflect the true likelihood of each class.
        :param model: The trained classification model.
        :param val_loader: DataLoader for the validation dataset.
        :param ema: Optional EMA object for applying smoothed weights during evaluation.
        :param init_t: Initial temperature value.
        :return: The optimized temperature parameter T.
        1. The optimized temperature T is a float value that scales the logits for better calibration.
        2. A T > 1 indicates softer probabilities, while T < 1 indicates peakier probabilities.
        3. The method uses LBFGS optimization to find the best T that minimizes the NLL loss on the validation set.
        """
        with no_grad():
            logits, y = self._collect_val_logits_targets(model, val_loader, ema)

        t = Parameter(tensor(float(init_t), device=self._device))
        nll = CrossEntropyLoss()

        # LBFGS optimizer for single parameter T, but could use others like Adam
        opt = LBFGS([t], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nll(logits / t.clamp_min(1e-3), y)  # keep T > 0
            loss.backward()
            return loss

        opt.step(closure)
        return float(t.detach().clamp_min(1e-3).cpu())

    def _search_logit_bias(self, model: PainTCNBiLSTMAttn, val_loader: DataLoader, y_val: ndarray) -> tuple[ndarray, float]:
        """
        Search for the best logit bias to maximize macro F1 score on validation data.

        Logit biasing involves adding a constant bias to the logits of each class before
        applying the softmax function. This technique can help adjust the decision boundaries
        of the classifier, potentially improving performance metrics such as the F1 score.
        :param model: The trained classification model.
        :param val_loader: DataLoader for the validation dataset.
        :param y_val: The true labels for the validation dataset.
        :return: Tuple containing the best logit bias array and the corresponding macro F1 score.
        1. best logit bias: An ndarray of shape (3,) representing the optimal bias for each class.
        2. corresponding macro F1 score: A float value indicating the highest macro F1 score achieved with the best bias.
        """
        logits_val = self._infer_logits_only(model, val_loader)
        grid = linspace(-0.6, 0.6, 13)
        best_f1, best_b = -1., zeros(3, dtype=numpy_float32)
        for b in product(grid, repeat=3):
            b = array(b, dtype=numpy_float32)
            pred = (logits_val + b[None, :]).argmax(1)
            f1 = f1_score(y_val, pred, average='macro', labels=self._classes.copy())
            if f1 > best_f1:
                best_f1, best_b = f1, b
        return best_b, best_f1

    @no_grad()
    def _evaluate_metrics(self, model: PainTCNBiLSTMAttn, val_loader: DataLoader) -> tuple[float, ndarray, ndarray]:
        """
        Evaluate macro F1 score, per-class recall, and confusion matrix on validation data.
        :param model: The trained classification model.
        :param val_loader: DataLoader for the validation dataset.
        :return: Tuple containing macro F1 score, per-class recall array, and confusion matrix.
        1. macro F1 score: A float value representing the overall F1 score across all classes.
        2. per-class recall array: An ndarray containing recall values for each class.
        3. confusion matrix: An ndarray representing the confusion matrix of predictions vs true labels.
        """
        model.eval()
        y_true, y_pred = [], []
        for _batch in val_loader:
            x_num, x_sta, x_surv, x_summ, y = self._get_data_from_batch(_batch)
            logits = model(x_num, x_surv, x_sta, x_summ)
            y_hat = logits.argmax(dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(y_hat.cpu().numpy())
        y_true = concatenate(y_true)
        y_pred = concatenate(y_pred)
        return (
            f1_score(y_true, y_pred, average='macro', labels=self._classes.copy()),
            recall_score(y_true, y_pred, average=None, labels=self._classes.copy()),
            confusion_matrix(y_true, y_pred, labels=self._classes.copy())
        )

    @no_grad()
    def _collect_val_logits_targets(
            self,
            model: PainTCNBiLSTMAttn,
            val_loader: DataLoader,
            ema: EMA | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Collect logits and targets from validation data.
        1. logits: A tensor of shape (N, C) where N is the number of samples and C is the number of classes.
        2. targets: A tensor of shape (N,) containing the true class labels.
        :param model: The trained classification model.
        :param val_loader: DataLoader for the validation dataset.
        :param ema: Optional EMA object for applying smoothed weights during evaluation.
        :return: Tuple containing logits tensor and targets tensor.
        """
        model.eval()
        if ema is not None:
            ema.apply_to(model)
        logits, ys = [], []
        for batch in val_loader:
            x_num, x_sta, x_surv, x_summ, y = self._get_data_from_batch(batch)
            logits.append(model(x_num, x_surv, x_sta, x_summ))
            ys.append(y)
        logits = cat(logits, dim=0)
        ys = cat(ys, dim=0)
        if ema is not None:
            ema.restore(model)
        return logits, ys

    @no_grad()
    def _infer_logits_only(self, model: PainTCNBiLSTMAttn, val_loader: DataLoader) -> ndarray:
        """
        Infer logits on validation data.

        The method runs the model in evaluation mode on the provided validation data loader
        and collects the output logits for all samples. The logits are detached from the computation
        graph and moved to the CPU before being converted to a NumPy array for further processing.
        :param model: The trained classification model.
        :param val_loader: DataLoader for the validation dataset.
        :return: An ndarray containing the inferred logits for all validation samples.
        """
        model.eval()
        outs = []
        for batch in val_loader:
            x_num, x_sta, x_surv, x_summ, y = self._get_data_from_batch(batch)
            logits = model(x_num, x_surv, x_sta, x_summ)
            outs.append(logits.detach().cpu().numpy())
        return concatenate(outs, axis=0)
