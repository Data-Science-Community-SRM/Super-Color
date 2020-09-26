import torch
import torch.nn as nn
import torch.optim as optim
import data_loader as dl
import matplotlib.pyplot as plt
import model as md

def train(model, train_loader, criterion, optimizer, num_epochs, inputShape):
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, _ in train_loader:
            # flattening n x n image into n^2 to feed into network
            # where n = inputShape
            images = images.view(inputShape, -1)

            # reinitializing gradients to zero to prevent exploding gradients
            optimizer.zero_grad()

            # making our predictions
            output = model(images)

            # calculating loss
            loss = criterion(output, labels)

            # backprop step
            loss.backward()

            running_loss += loss.item()
            losses.append(running_loss)

            optimizer.step()

    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

def main():
    path = "."

    batch_size = 8
    train_loader = dl.get_data(path, batch_size)
    
    inputShape = ..
    model = md.AutoEncoder(inputShape)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 500
    
    train(model, train_loader, criterion, optimizer, num_epochs, inputShape)

    img_number = 4
    model.eval()

    with torch.no_grad():
        test_output = model(img_number)
        fig = plt.figure()
        plt.subplot(.., ..)
        plt.imshow(test_output)

if __name__ == "__main__":
    main()