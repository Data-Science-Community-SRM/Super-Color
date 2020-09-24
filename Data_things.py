
"""
All data manupilation things in this file

"""

import torchvision
import torch.utils.data

# change name according to main
def get_data( loc ):
    """
    Simple function that gives us torchvision dataset.

    This is the most basic version. We will have to make it more complex as we go


    """
    x = torchvision.datasets.ImageFolder(loc,torchvision.transforms.CenterCrop(250))
    return x

if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data related things")
    
