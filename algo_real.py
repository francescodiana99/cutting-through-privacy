import torch
import torchvision
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import itertools

import matplotlib.pyplot as plt

import bisect
import random
import math
import argparse
import os

from utils_search import *
from utils_misc import *
from utils_test import * 

def main():

    args = parse_args()

    set_seeds(3)

    images, all_labels = prepare_data()

    num_samples = args.n_samples
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    



if __name__ == '__main__':
    main()