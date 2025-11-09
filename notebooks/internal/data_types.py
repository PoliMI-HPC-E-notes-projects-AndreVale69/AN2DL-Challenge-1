"""
Data types for storing datasets and related information for pain intensity classification.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class LabelMap(dict[str, int]):
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
class DataSet(dict):
    """
    Data structure for storing training, validation, and test datasets along with scalers and label mappings.

    Attributes:
        X_dyn_train (np.ndarray): Dynamic features for training.
        X_sta_train (pd.DataFrame): Static features for training.
        y_train (np.ndarray): Labels for training.
        X_dyn_val (np.ndarray): Dynamic features for validation.
        X_sta_val (pd.DataFrame): Static features for validation.
        y_val (np.ndarray): Labels for validation.
        X_dyn_test (np.ndarray): Dynamic features for testing.
        X_sta_test (np.ndarray): Static features for testing.
        ids_test (np.ndarray): Identifiers for test samples.
        class_weights (dict[int, float]): Class weights for handling class imbalance.
        scaler_dyn (StandardScaler): Scaler for dynamic features.
        scaler_sta (StandardScaler): Scaler for static features.
        label_map (LabelMap): Mapping of pain intensity levels to integer labels.
    """
    X_dyn_train: np.ndarray
    X_sta_train: pd.DataFrame
    y_train: np.ndarray
    X_dyn_val: np.ndarray
    X_sta_val: pd.DataFrame
    y_val: np.ndarray
    X_dyn_test: np.ndarray
    X_sta_test: np.ndarray
    ids_test: np.ndarray
    class_weights: dict[int, float]
    scaler_dyn: StandardScaler
    scaler_sta: StandardScaler
    label_map: Union[LabelMap, dict[str, int]]