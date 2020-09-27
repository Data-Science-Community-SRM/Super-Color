import torch
import torch.nn as nn
import torch.optim as optim
import Data_things as dl
import matplotlib.pyplot as plt
import model as md
from torch.utils.data import Dataset,DataLoader

def train(model, train_loader, criterion, optimizer, num_epochs, inputShape):
    losses = []
    for _ in range(num_epochs):
        running_loss = 0.0

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
            print(loss.item())
            optimizer.step()

    # fig = plt.figure()
    # plt.plot(losses)
    # plt.xlabel('Epochs')
    # plt.ylabel('Losses')

def main():
    path = "."
    batch_size = 2

    dataset = dl.REcolorDataset(path)
    train_loader = DataLoader(dataset,batch_size)
    
    inputShape = 4
    model = md.AutoEncoder(inputShape)

    A = dataset[0][0].unsqueeze(0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 20
    
    train(model, train_loader, criterion, optimizer, num_epochs, inputShape)
    """
    img_number = 4
    model.eval()

    with torch.no_grad():
        test_output = model(img_number)
        fig = plt.figure()
        plt.subplot(.., ..)
        plt.imshow(test_output)
    """

if __name__ == "__main__":
    main()