import torch
from skimage.io import imread
import matplotlib.pyplot as plt

class WeatherDataset(torch.utils.data.Dataset):

    def __init__(self, lists, transform=None):
        assert len(lists[0]) == len(lists[1])
        self.lists = lists
        self.transform = transform

    def __getitem__(self, index):
        x = imread(self.lists[0][index])
        if self.transform:
            x = self.transform(x)
        y = self.lists[1][index]
        return x, y

    def __len__(self):
        return len(self.lists[0])

    def imshow(self, index):
        x = imread(self.lists[0][index])
        if self.transform:
            x = self.transform(x)
        plt.imshow(x)
        plt.show()
