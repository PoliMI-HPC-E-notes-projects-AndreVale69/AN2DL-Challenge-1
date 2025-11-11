"""
Utility class for persisting datasets and scalers using joblib.
"""
import os
from typing import Literal, Union

import joblib

from internal.data_types import DataSet, DataSetV2


class PersistenceManager:
    """
    Manages the persistence of datasets and scalers using joblib.

    Attributes:
        ARRAYS_AND_SCALERS_PATH (str): Path to store/load arrays and scalers.
        ARRAYS_V2_PATH (str): Path to store/load version 2 arrays.
    """
    # full path to the current directory
    NOTEBOOKS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARRAYS_AND_SCALERS_PATH = f'{NOTEBOOKS_PATH}/processed/arrays_and_scalers.joblib'
    ARRAYS_V2_PATH = f'{NOTEBOOKS_PATH}/processed/arrays_v2.joblib'

    @staticmethod
    def _makedirs_for_file(file_path: str) -> None:
        """
        Ensures that the directory for the given file path exists.
        :param file_path: The file path for which to create directories.
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def load_arrays_and_scalers() -> DataSet:
        """
        Loads the dataset containing arrays and scalers from the specified path.
        :return: DataSet object with loaded data.
        """
        PersistenceManager._makedirs_for_file(PersistenceManager.ARRAYS_AND_SCALERS_PATH)
        return (
            print(f"Arrays and scalers loaded successfully from: {PersistenceManager.ARRAYS_AND_SCALERS_PATH}") or
            DataSet(**joblib.load(PersistenceManager.ARRAYS_AND_SCALERS_PATH))
        )

    @staticmethod
    def save_arrays_and_scalers(data: Union[DataSet, dict], compress: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]=0) -> None:
        """
        Saves the dataset containing arrays and scalers to the specified path.
        :param data: DataSet object to be saved.
        :param compress: Compression level for joblib dump (0-9). Default is 0 (no compression).
        """
        PersistenceManager._makedirs_for_file(PersistenceManager.ARRAYS_AND_SCALERS_PATH)
        joblib.dump(
            dict(data) if isinstance(data, DataSet) else data,
            PersistenceManager.ARRAYS_AND_SCALERS_PATH,
            compress=compress
        )
        print(f"Arrays and scalers saved successfully to: {PersistenceManager.ARRAYS_AND_SCALERS_PATH}")

    @staticmethod
    def load_arrays_v2() -> DataSetV2:
        """
        Loads the version 2 dataset containing arrays from the specified path.
        :return: DataSet object with loaded data.
        """
        PersistenceManager._makedirs_for_file(PersistenceManager.ARRAYS_V2_PATH)
        return (
            print(f"Arrays v2 loaded successfully from: {PersistenceManager.ARRAYS_V2_PATH}") or
            DataSetV2(**joblib.load(PersistenceManager.ARRAYS_V2_PATH))
        )

    @staticmethod
    def save_arrays_v2(data: Union[DataSetV2, dict], compress: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]=0) -> None:
        """
        Saves the version 2 dataset containing arrays to the specified path.
        :param data: DataSet object to be saved.
        :param compress: Compression level for joblib dump (0-9). Default is 0 (no compression).
        """
        PersistenceManager._makedirs_for_file(PersistenceManager.ARRAYS_V2_PATH)
        joblib.dump(
            dict(data) if isinstance(data, DataSetV2) else data,
            PersistenceManager.ARRAYS_V2_PATH,
            compress=compress
        )
        print(f"Arrays v2 saved successfully to: {PersistenceManager.ARRAYS_V2_PATH}")