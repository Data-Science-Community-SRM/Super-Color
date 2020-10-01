import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Data_things as dl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as trs
from tqdm import tqdm
import numpy as np


class REcolorDataset(Dataset):
    def __init__(self,loc):
        self.x = torchvision.datasets.ImageFolder(loc)

    def __len__(self):
        return self.x.__len__()

    def __getitem__(self,idx):
        inp = trs.Compose([trs.RandomCrop(128)])
        grey = trs.Grayscale(1)
        tt = trs.Compose([trs.ToTensor(),trs.Normalize(mean=0.5,std =0.5)])

        img = inp(self.x[idx][0])
        return tt(grey(img)) ,tt(img)


# change name according to main
def normlocal(img):
    return img

if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data_things test")
    x = getdataset(".")

    i = x[4]
    exit()
    plt.imshow(i[0])
    plt.show()
    plt.imshow(i[1])
    plt.show()

    
