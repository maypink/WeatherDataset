import os
from skimage.io import imread
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import cv2
from sklearn.preprocessing import LabelEncoder
import random

def reprod_init():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

def get_data():
    x_data = []
    y_data = []
    for (dirpath, dirname, filenames) in os.walk('dataset2'):
        for filename in filenames:
            if filename.endswith('jpg'):
                label = ''
                for i in filename:
                    if i.isalpha():
                        label += i
                    else:
                        break
                assert label in ['sunrise', 'rain', 'cloudy', 'shine']
                filepath = os.path.join(dirpath, filename)
                image = imread(filepath)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    x_data.append(filepath)
                    y_data.append(label)

    y_data[:] = LabelEncoder().fit_transform(y_data[:])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)
    return x_train, x_test, y_train, y_test


def resize(x):
    x = cv2.resize(x, (64, 64))
    return x


def to_tensor(x):
    x=torch.tensor(x)
    return x