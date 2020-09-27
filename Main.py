import torch
import torch.nn as nn
import torch.optim as optim
import Data_things as dl
import matplotlib.pyplot as plt
import model as md
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import numpy as np

def train(model, train_loader, criterion, optimizer, num_epochs, inputShape):
    losses = [0,0,0]
    for e in range(num_epochs):
        running_loss = 0.0
        print(e)
        i =0
        for images, out in train_loader:
            # flattening n x n image into n^2 to feed into network
            # where n = inputShape

            # reinitializing gradients to zero to prevent exploding gradients
            optimizer.zero_grad()

            # making our predictions
            output = model(images)

            # calculating loss
            loss = criterion(output,out)

            # backprop step
            loss.backward()

            running_loss += loss.item()
            losses.append(running_loss)
            print(losses[-2]-losses[-1])
            if i % 200 == 0:
                Z = output.detach()
                X = images.detach()
                plt.imshow(dl.Normalizepls(np.transpose(Z[0],(1,2,0))))
                plt.show()
                plt.imshow(dl.Normalizepls(X[0])[0])
                plt.show()
            i += 1
            optimizer.step()



    # fig = plt.figure()
    # plt.plot(losses)
    # plt.xlabel('Epochs')
    # plt.ylabel('Losses')

def main():
    path = "."
    batch_size = 1

    dataset = dl.REcolorDataset(path)
    print(len(dataset))
    train_loader = DataLoader(dataset,batch_size)
    
    inputShape = 4
    model = md.AutoEncoder(inputShape)

    A = dataset[0][0].unsqueeze(0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100
    
    train(model, train_loader, criterion, optimizer, num_epochs, inputShape)


if __name__ == "__main__":
    main()