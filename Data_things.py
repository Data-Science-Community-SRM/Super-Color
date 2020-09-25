
"""
All data manupilation things in this file

"""

import torchvision
import torch.utils.data
import torchvision.transforms as trs

# change name according to main
def get_data( loc ):
    """
    Simple function that gives us torchvision dataset.

    This is the most basic version. We will have to make it more complex as we go


    """
    y = trs.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
    x = torchvision.datasets.ImageFolder(loc,y)
    return x
    
if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data related things")
    
