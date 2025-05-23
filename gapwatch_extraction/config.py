###########################################################################################################
# Libraries
###########################################################################################################

# General purpose 
import rosbag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NMF implementation
from sklearn.decomposition import NMF

# Autoencoder implementation
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# PCA implementation
from sklearn.decomposition import PCA

