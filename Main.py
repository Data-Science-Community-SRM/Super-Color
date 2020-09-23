import data_loader as dl
import matplotlib.pyplot as plt

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
    for img, _ in dataloader:
        plt.imshow(img[0])
        plt.show()

    
# This will call main function only when this is the main file
# when you import a python script it is not the main file
if __name__ == "__main__":
    main()