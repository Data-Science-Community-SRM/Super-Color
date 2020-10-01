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
import model as md


def train(model, train_loader, criterion, optimizer, num_epochs, inputShape):
    losses = [0,0,0]
    for e in range(num_epochs):
        running_loss = 0.0
        i =0
        for images, out in train_loader:
            # flattening n x n image into n^2 to feed into network
            # where n = inputShape

            # reinitializing gradients to zero to prevent exploding gradients
            optimizer.zero_grad()

            # making our predictions
            output = model(images)

            # calculating loss
            #loss2 =  torch.mean(2 * torch.log(torch.empty(output.shape).fill_(1)) - torch.log(output-out))
            #loss = -1 * torch.dot(output.flatten(),out.flatten())
            
            #loss = criterion(sqnetFET(out),sqnetFET(output))

            loss = criterion(out,output)
            # backprop step
            loss.backward()

            running_loss += loss.item()
            losses.append(running_loss)
            if i % 200 == 0:
                Z = output.detach()
                X = images.detach()
                fig,a =  plt.subplots(1,3,sharex=True,sharey=True,)
                a[0].imshow(X[0][0],cmap='gray')
                a[1].imshow(np.transpose(Z[0],(1,2,0)))
                a[2].imshow(np.transpose(out[0],(1,2,0)))
                a[0].axis(False)
                a[1].axis(False)
                a[2].axis(False)
                a[0].title.set_text("Input")
                a[1].title.set_text("Output")
                a[2].title.set_text("GroundTruth")
                plt.show()
                plt.plot(losses)
                plt.show()
            i += 1
            optimizer.step()



    # fig = plt.figure()
    # plt.plot(losses)
    # plt.xlabel('Epochs')
    # plt.ylabel('Losses')

def main():
    path = "."
    batch_size = 8

    dataset = dl.REcolorDataset(path)
    train_loader = DataLoader(dataset,batch_size)
    
    inputShape = 4
    model = md.AutoEncoder(inputShape)

    A = dataset[0][0].unsqueeze(0)
    print(torch.min(A),torch.max(A),A)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 5
    
    train(model, train_loader, criterion, optimizer, num_epochs, inputShape)

if __name__ == "__main__":
    main()