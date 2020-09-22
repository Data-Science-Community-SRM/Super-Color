"""
Main file for compiling everything

training, inference everything will be run from here 

"""

# importing our personal file 
import Data_things


import matplotlib.pyplot as plt



def main():
    """
    can only show image for now
    """
    print("Running main now")

    # getting images from this directory
    z = Data_things.get_data(".")
    
    print(z)
    print("Got data")

    # print first image
    plt.imshow(z[4][0])
    plt.grid(False)
    plt.show()

    
# This will call main function only when this is the main file
# when you import a python script it is not the main file
if __name__ == "__main__":
    main()