"""This file implements the UCI Human Activity Recognition Using Smartphines dataset."""

import logging
import os
import numpy as np
import pandas as pd
import zipfile
import ssl

from torch.utils.data import Dataset, DataLoader
import urllib

from .constants import URL

class HARUSDataset(Dataset):
    """PyTorch Dataset class for the Human Activity Recognition Using Smartphines dataset.
    
    Args:
        cache_dir(str): Directory to store the dataset.
        download(bool): Whether to download the dataset. Default is True.
        mode(str): Mode of the dataset. Default is 'train'.
        use_double_precision(bool): Whether to use double precision. Default is False.
    """

    def __init__(self, cache_dir: str='./data/harus', mode='train', download: bool=True):
        """
        Raises:

        """

        self.cache_dir = cache_dir
        self.download = download
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
        self.targets = self.dataset['targets'].values
        self.features = self.dataset.drop(columns=['targets']).values
    

    def _download_and_preprocess(self):
        """
        Download and preprocess the dataset.
        """
        try:

            ssl._create_default_https_context = ssl._create_unverified_context 
            
            import zipfile

            os.makedirs(os.path.join(self.cache_dir, 'raw'))

            zip_file_path, _ =  urllib.request.urlretrieve(URL, os.path.join(self.cache_dir, 'raw', 'harus.zip'))

        except urllib.error.URLError:
            logging.error("Could not download the dataset. Please check your internet connection.")
            raise urllib.error.URLError
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.cache_dir, 'raw'))


        with zipfile.ZipFile(os.path.join(self.cache_dir, 'raw', 'UCI HAR Dataset.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.cache_dir, 'raw'))

        with open(os.path.join(self.cache_dir, 'raw', 'UCI HAR Dataset', 'train', 'X_train.txt')) as f:
            train_data = pd.read_csv(f, header=None, sep=r'\s+')
        with open(os.path.join(self.cache_dir, 'raw', 'UCI HAR Dataset', 'train', 'y_train.txt')) as f:
            train_labels = pd.read_csv(f, header=None)
        with open(os.path.join(self.cache_dir, 'raw', 'UCI HAR Dataset', 'test', 'X_test.txt')) as f:
            test_data = pd.read_csv(f, header=None, sep=r'\s+')
        with open(os.path.join(self.cache_dir, 'raw', 'UCI HAR Dataset', 'test', 'y_test.txt')) as f:
            test_labels = pd.read_csv(f, header=None)
        
        train_labels.rename(columns={0: 'targets'}, inplace=True)
        train_labels['targets'] = train_labels['targets'] - 1
        test_labels.rename(columns={0: 'targets'}, inplace=True)
        test_labels['targets'] = test_labels['targets'] - 1

        train_dataset = pd.concat([train_data, train_labels], axis=1)
        test_dataset = pd.concat([test_data, test_labels], axis=1)

        os.makedirs(os.path.join(self.cache_dir, 'processed'), exist_ok=True)

        train_dataset.dropna(inplace=True)
        test_dataset.dropna(inplace=True)

        train_dataset.to_csv(os.path.join(self.cache_dir, 'processed', 'train.csv'), index=False)
        test_dataset.to_csv(os.path.join(self.cache_dir, 'processed', 'test.csv'), index=False)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    

    def get_dataset(self):
        """
        Returns the dataset.
        
        Retu
            pd.DataFrame: Dataset.rns:
        """

        return self.dataset
    
    def get_column_names(self):
        """
        Returns the column names of the dataset.
        
        Returns:
            list: Column names.
        """
        return self.column_names
    
    def get_targets(self):
        """
        Returns the targets of the dataset.
        
        Returns:
            np.ndarray: Targets.
        """
        return self.targets
    
    def get_features(self):
        """
        Returns the features of the dataset.
        
        Returns:
            np.ndarray: Features.
        """
        return self.features
    

if __name__ == '__main__':

    HARUSDataset()
