import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

import Data_things as dl
import model as md
import train as tr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = './Images/'
    batch_size = 1

    print("Started Training")
    dataset = dl.REcolorDataset(path)
    train_loader = DataLoader(dataset,batch_size)
    
    model = md.AutoEncoder()

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 50
    
    model = model.to(device)

    tr.train(model, train_loader, criterion, optimizer, num_epochs,device)

    torch.save(model, ".temp")

if __name__ == "__main__":
    main()