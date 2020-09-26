import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as trs

import matplotlib.pyplot as plt


class REcolorDataset(Dataset):
    def __init__(self,loc):
        self.x = torchvision.datasets.ImageFolder(loc)

    def __len__(self):
        return self.x.__len__()

    def __getitem__(self,idx):
        inp = trs.Compose([trs.RandomCrop(250),trs.ToTensor()])
        out = trs.Compose([trs.Grayscale(1)])
        img = inp(self.x[idx][0])
        return out(img), img


# change name according to main
def getdataset( loc ):
    """
    Simple function that gives us torchvision dataset.

    This is the most basic version. We will have to make it more complex as we go


    """
    x = REcolorDataset(loc)
    return x

def getdataloader(dataset, bs):
    return DataLoader(dataset,bs)


if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data_things test")
    x = getdataset(".")
    print(x[1])
    i = x[4]
    plt.imshow(i[0])
    plt.show()
    plt.imshow(i[1])
    plt.show()

    
