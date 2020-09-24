import data_loader as dl
import matplotlib.pyplot as plt


def train(model, train_loader):
    losses = []

    for epoch in range(100):
        running_loss = 0.0

        for images, labels in train_loader:
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
    print("Running main now")

    # getting images from this directory
    train_loader, test_loader = dl.get_data(".")
    
    # print(z)
    # print("Got data")

    # print first image
    # plt.imshow(z[4][0])
    # plt.grid(False)
    # plt.show()

    # I think these two lines should probably print the first image
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    for num in range(1, 6):
        plt.subplot(10, 10, num)
        plt.axis('off')
        plt.imshow(images[num].numpy().squeeze())



    
# This will call main function only when this is the main file
# when you import a python script it is not the main file
if __name__ == "__main__":
    main()