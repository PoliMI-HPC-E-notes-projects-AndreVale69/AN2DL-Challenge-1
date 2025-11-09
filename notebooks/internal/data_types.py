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
class FeatureEngineeringConfig:
    """
    Configuration for feature engineering options.

    Attributes:
        delta (bool): Whether to compute delta features.
        rolling_std (bool): Whether to compute rolling standard deviation features.
        window (int): Window size for rolling calculations.
    """
    delta: bool
    rolling_std: bool
    window: int

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
        feat_eng (FeatureEngineeringConfig | dict): Feature engineering configuration.
    """
    X_dyn_num_train: np.ndarray
    X_dyn_num_val: np.ndarray
    X_dyn_num_test: np.ndarray
    X_surv_train: list[np.ndarray]
    X_surv_val: list[np.ndarray]
    X_surv_test: list[np.ndarray]
    feat_eng: Union[FeatureEngineeringConfig, dict[str, bool | int]]
