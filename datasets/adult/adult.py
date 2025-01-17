import os
import shutil
import ssl
import json
import urllib
import logging

from .constants import *

import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class AdultDataset(Dataset):
    """
    PyTorch Dataset class for the Adult dataset.

    Args:
        cache_dir(str): Directory to store the dataset.
        scaler_name(str): Name of the scaler to use. Default is 'standard'.
        download(bool): Whether to download the dataset. Default is True.
        seed(int): Seed for reproducibility. Default is 42.
        use_double_precison(bool): Whether to use double precision. Default is False.
        mode(str): Mode of the dataset. Default is 'train'.
    """

    def __init__(self, cache_dir: str='./data/adult', scaler_name: str='standard', download: bool=True, seed: int=42, mode: str='train'):
        """
        Raises:

        """

        self.cache_dir = cache_dir
        self.scaler_name = scaler_name
        self.download = download
        self.seed = seed
        self.mode = mode

        if self.download:
            logging.info("Downloading and pre-processing the dataset...")
            self._download_and_preprocess()
            logging.info("Dataset downloaded and pre-processed. Loading data...")
            self._load_dataset()

        else:
            if not os.path.exists(self.cache_dir):
                raise FileNotFoundError(
                    "Data directory does not exist. Please set download to True to download the data."
                    )
            else:
                self._load_dataset()


    def _load_dataset(self):
        """
        Load the dataset from the cache directory.
        """

        if self.mode not in ['train', 'test']:
            raise ValueError("Mode should be either 'train' or 'test'.")
        
        self.dataset = pd.read_csv(os.path.join(self.cache_dir, 'processed', f'{self.mode}.csv'))
        self.column_names = self.dataset.columns
        self.features = self.dataset.drop(columns=['income']).values
        self.targets = self.dataset['income'].values

    
    def _download_and_preprocess(self):
        """
        Download the dataset and preprocess it.

        The pre-processing involves the following steps:
            - Drop columns with missing values.
            - Drop 'fnlwgt' column.
            - Replace 'income' column with binary values.
            - Get dummy variables for categorical features.
            - Scale the data.
        """

        try:
            train_df = pd.read_csv(TRAIN_URL, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
            test_df = pd.read_csv(TEST_URL, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?", skiprows=1)

        except urllib.error.URLError:

            ssl._create_default_https_context = ssl._create_unverified_context

            import zipfile

            os.makedirs(os.path.join(self.cache_dir, 'raw'), exist_ok=True)

            zip_file_path, _ = urllib.request.urlretrieve(BACKUP_URL, os.path.join(self.cache_dir, 'raw', 'adult.zip'))

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.cache_dir, 'raw'))

            train_path = os.path.join(self.cache_dir, 'raw', "adult.data")
            test_path = os.path.join(self.cache_dir, 'raw', "adult.test")

            train_df = pd.read_csv(train_path, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
            test_df = pd.read_csv(test_path, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?", skiprows=1)

        if self.cache_dir is not None:
            os.makedirs(os.path.join(self.cache_dir, 'raw'), exist_ok=True)

            raw_train_path = os.path.join(self.cache_dir, 'raw', 'train.csv')
            train_df.to_csv(raw_train_path, index=False)
            logging.debug(f"Raw train data cached at: {raw_train_path}")

            raw_test_path = os.path.join(self.cache_dir, 'raw', 'test.csv')
            test_df.to_csv(raw_test_path, index=False)
            logging.debug(f"Raw test data cached at: {raw_test_path}")

        train_df = train_df.drop(columns=['fnlwgt'])
        test_df= test_df.drop(columns=['fnlwgt'])


        train_df['income'] = train_df['income'].replace('<=50K', 0).replace('>50K', 1)
        test_df['income'] = test_df['income'].replace('<=50K.', 0).replace('>50K.', 1)

        train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)
        test_df = pd.get_dummies(test_df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        self.scaler = self.set_scaler(self.scaler_name)

        train_df = self._scale_features(train_df, self.scaler, mode="train")
        #TODO: decide how to handle this case (categorical value missing in the training set raising error)
        # test_df = self._scale_features(test_df, self.scaler, mode="test")

        if self.cache_dir is not None:
            os.makedirs(os.path.join(self.cache_dir, 'processed'), exist_ok=True)
            processed_train_path = os.path.join(self.cache_dir, 'processed', 'train.csv')
            processed_test_path = os.path.join(self.cache_dir, 'processed', 'test.csv')
            train_df.to_csv(processed_train_path, index=False)
            test_df.to_csv(processed_test_path, index=False)
            logging.debug(f"Processed train data cached at: {processed_train_path}")
            logging.debug(f"Processed test data cached at: {processed_test_path}")

        return train_df, test_df
    

    def set_scaler(self, scaler_name):
        """
        Set the scaler to use for the dataset.

        Args:
            scaler_name(str): Name of the scaler to use. Default is 'standard'.

        Returns:
            Scaler: Scaler to use for the dataset.
        """

        if scaler_name == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaler_name == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif scaler_name == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError("Scaler not implemented.")

        return scaler
    

    @staticmethod
    def _scale_features(df, scaler, mode="train"):
        numerical_columns = df.select_dtypes(include=['number']).columns

        numerical_columns = numerical_columns[numerical_columns != 'income']

        income_col = df['income']
        
        age_col = df['age']

        features_numerical = df[numerical_columns]

        if mode == "train":
            features_numerical_scaled = \
                pd.DataFrame(scaler.fit_transform(features_numerical), columns=numerical_columns)
        else:
            features_numerical_scaled = \
                pd.DataFrame(scaler.transform(features_numerical), columns=numerical_columns)

        features_scaled = pd.concat([age_col, features_numerical_scaled, income_col], axis=1)

        return features_scaled
    
    
    def get_dataset(self):
        """
        Returns the dataset.
        
        Returns:
            pd.DataFrame: Dataset.
        """

        return self.dataset
    

    def __len__(self):
        """
        Returns the number of samples in the dataset
        
        Returns:
            int: Number of samples in the dataset
        """

        return len(self.dataset)


    def __getitem__(self, idx: int):
        """
        Return a tuple representing the idx-th sample in the dataset.
        Args:
            idx(int): Index of the sample to return.
            
        Returns:
            Tuple(torch.LongTensor, torch.Tensor): Tuple containing the features and the target.
        """

        return torch.Tensor(self.features[idx], int(self.targets[idx]))
