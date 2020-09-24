import data_loader as dl
import matplotlib.pyplot as plt


# importing our personal file 
import Data_things
import model

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
    """
    Can call function from now
    """
    #get images from  current folder
    path = "."
    dataset = Data_things.getdataset(path)
    #wrap datset into batchsz
    batchsz = 8
    datalaoder = Data_things.getdataloader(batchsz)
    # make instance of model class
    m = model.Autoenc()
    #train instance m
    model.train(m) # shoul have the training loop 

    # check prfromance on image 4
    img_number = 4
    model.evalu(m,img_number)


    
# This will call main function only when this is the main file
# when you import a python script it is not the main file
if __name__ == "__main__":
    main()