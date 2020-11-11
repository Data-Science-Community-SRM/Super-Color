import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as trans
from tqdm import tqdm
import numpy as np
from os import walk, path, mkdir,listdir
from skimage import io, color
from PIL import Image

class REcolorDataset(Dataset):
    def __init__(self,loc = "./val_256"):
        self.root = loc
        self.names = []
        self.counter = 0

        self.transforms = trans.Compose([trans.Resize(256), trans.CenterCrop(256), trans.ToTensor()])
        
        for (dirpath, dirnames, filenames) in walk(self.root):
            self.names.append([dirpath] + filenames)
        
        if not path.exists("./temp"):
            mkdir("./temp")
        
        if not listdir("./temp"):
            print("Caching files")
            for files in tqdm(self.names):
                for i in range(1,len(files)):
                    im = io.imread(files[0]+"/"+files[i])
                    if len(im.shape) == 3:
                        lab = (((color.rgb2lab(im)+[0,128,128])/ [100, 255, 255])*255).astype(np.uint8)
                        im = Image.fromarray(lab)
                        im = self.transforms(im)
                        torch.save(im,"./temp/"+str(self.counter))
                        self.counter += 1
        else:
            count = max([int(f) for f in listdir("./temp/")])
            self.counter = count
    
    def __len__(self):
        return self.counter
        
    def __getitem__(self,idx): 
        im = torch.load("./temp/"+str(idx))
        return im[0].unsqueeze(0),im[1:]

if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data_things test")
    x = REcolorDataset("./Images")
    print(len(x))

    
