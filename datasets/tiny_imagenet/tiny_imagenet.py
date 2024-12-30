import os
from torchvision.datasets import ImageFolder


class TinyImageNetDataset:
    """
    Dataset Class for Tiny Imagenet dataset

    Args:
        root(str): Path to the root folder of the Tiny ImageNet dataset. 
        def
    """
    def __init__(self, root, transform):
        self.root = os.path.expanduser(root)   
        self.transform = transform
        
        self.train_dataset = self._get_train_dataset(self.transform)
        self.val_dataset = self._get_val_dataset(self.transform)
    
    def _get_train_dataset(self, transform):
        train_dir = os.path.join(self.root)
        return ImageFolder(train_dir, transform=transform) 
    

    def _get_val_dataset(self, transform):
        val_dir = os.path.join(self.root)
        return ImageFolder(val_dir, transform=transform)
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset