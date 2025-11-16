"""
Data types for storing datasets and related information for pain intensity classification.
"""
from dataclasses import dataclass, asdict
from typing import Union

from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, Tensor
from torch.utils.data import Dataset


@dataclass
class DictLike:
    """
    A base class that mimics dictionary behavior for dataclasses.
    """
    def to_dict(self) -> dict:
        return asdict(self)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.to_dict())

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __contains__(self, key):
        return key in self.to_dict()

    def __or__(self, other):
        if isinstance(other, dict):
            return {**self.to_dict(), **other}
        return NotImplemented

@dataclass
class LabelMap(DictLike):
    """
    Mapping of pain intensity levels to integer labels.

    Attributes:
        no_pain (int): Label for no pain.
        low_pain (int): Label for low pain.
        high_pain (int): Label for high pain.
    """
    no_pain: int
    low_pain: int
    high_pain: int

@dataclass
class FeatureEngineeringConfig(DictLike):
    """
    Configuration for feature engineering options.

    Attributes:
        delta (bool): Whether to compute delta features.
        rolling_std (bool): Whether to compute rolling standard deviation features.
        seq_summaries (bool): Whether to compute sequence-level summary statistics.
        window (int): Window size for rolling calculations.
    """
    delta: bool
    rolling_std: bool
    seq_summaries: bool
    window: int

@dataclass
class DataSet(DictLike):
    """
    Data structure for storing training, validation, and test datasets along with scalers and label mappings.

    Attributes:
        X_dyn_train (np.ndarray): Dynamic features for training.
        X_sta_train (pd.ndarray): Static features for training.
        y_train (np.ndarray): Labels for training.
        X_dyn_val (np.ndarray): Dynamic features for validation.
        X_sta_val (pd.ndarray): Static features for validation.
        y_val (np.ndarray): Labels for validation.
        X_dyn_test (np.ndarray): Dynamic features for testing.
        X_sta_test (np.ndarray): Static features for testing.
        pid_tr_seq (np.ndarray): Patient IDs for training sequences.
        pid_train (np.ndarray): Identifiers for training samples.
        pid_val (np.ndarray): Identifiers for validation samples.
        ids_test (np.ndarray): Identifiers for test samples.
        class_weights (dict[int, float]): Class weights for handling class imbalance.
        scaler_dyn (StandardScaler): Scaler for dynamic features.
        scaler_sta (StandardScaler): Scaler for static features.
        label_map (LabelMap): Mapping of pain intensity levels to integer labels.
    """
    X_dyn_train: ndarray
    X_sta_train: ndarray
    y_train: ndarray
    X_dyn_val: ndarray
    X_sta_val: ndarray
    y_val: ndarray
    X_dyn_test: ndarray
    X_sta_test: ndarray
    pid_tr_seq: ndarray
    pid_train: ndarray
    pid_val: ndarray
    ids_test: ndarray
    class_weights: dict[int, float]
    scaler_dyn: StandardScaler
    scaler_sta: StandardScaler
    label_map: Union[LabelMap, dict[str, int]]

@dataclass
class DataSetV2(DataSet):
    """
    Extended data structure for storing numerical dynamic features and feature engineering configuration.
    Inherits from DataSet and adds additional attributes.

    Attributes:
        X_dyn_num_train (np.ndarray): Numerical dynamic features for training.
        X_dyn_num_val (np.ndarray): Numerical dynamic features for validation.
        X_dyn_num_test (np.ndarray): Numerical dynamic features for testing.
        X_surv_train (list[np.ndarray]): List of survival analysis features for training.
        X_surv_val (list[np.ndarray]): List of survival analysis features for validation.
        X_surv_test (list[np.ndarray]): List of survival analysis features for testing.
        X_dyn_summ_train (np.ndarray): Sequence-level summary statistics for training.
        X_dyn_summ_val (np.ndarray): Sequence-level summary statistics for validation.
        X_dyn_summ_test (np.ndarray): Sequence-level summary statistics for testing.
        scaler_dyn_summ (StandardScaler): Scaler for sequence-level summary statistics.
        feat_eng (FeatureEngineeringConfig | dict): Feature engineering configuration.
    """
    X_dyn_num_train: ndarray
    X_dyn_num_val: ndarray
    X_dyn_num_test: ndarray
    X_surv_train: list[ndarray]
    X_surv_val: list[ndarray]
    X_surv_test: list[ndarray]
    # Sequence-level summaries statistics (mean, std, min, max, etc.)
    X_dyn_summ_train: ndarray
    X_dyn_summ_val: ndarray
    X_dyn_summ_test: ndarray
    scaler_dyn_summ: StandardScaler
    # feature engineering configuration
    feat_eng: Union[FeatureEngineeringConfig, dict[str, bool | int]]

    def __post_init__(self):
        # Ensure that feat_eng is an instance of FeatureEngineeringConfig
        if isinstance(self.feat_eng, dict):
            self.feat_eng = FeatureEngineeringConfig(**self.feat_eng)

class PainDataset(Dataset):
    """
    PyTorch Dataset for pain intensity classification.

    It handles dynamic numerical features, survival analysis features, static features,
    sequence-level summary statistics, and optional labels.

    It is designed to be compatible with PyTorch's DataLoader for efficient batching and shuffling.
    """
    def __init__(
            self,
            x_dyn_num: ndarray,
            x_surv: list[ndarray],
            x_sta: ndarray,
            x_summ: ndarray,
            y: ndarray | None = None,
    ):
        """
        PyTorch Dataset for pain intensity classification.
        :param x_dyn_num: The dynamic numerical features.
        :param x_surv: The survival analysis features.
        :param x_sta: The static features.
        :param x_summ: The sequence-level summary statistics.
        :param y: The labels (optional).
        """
        self.X_dyn_num: Tensor      = from_numpy(x_dyn_num).float()
        self.X_sta: Tensor          = from_numpy(x_sta).float()
        self.X_surv: list[Tensor]   = [from_numpy(s).long() for s in x_surv]
        self.x_summ: Tensor         = from_numpy(x_summ).float()
        self.y: Tensor | None       = from_numpy(y).long() if y is not None else None

    def __len__(self) -> int:
        return self.X_dyn_num.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = {
            'x_num':  self.X_dyn_num[idx],
            'x_surv': [s[idx] for s in self.X_surv],
            'x_sta':  self.X_sta[idx],
            'x_summ': self.x_summ[idx],
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        return item
