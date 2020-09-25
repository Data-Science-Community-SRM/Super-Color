import torchvision
import torch.utils.data
import torchvision.transforms as trs

def get_data(loc):
    y = trs.Compose([trs.CenterCrop(250), trs.Grayscale(num_output_channels=1)])
    
    x = torchvision.datasets.ImageFolder(loc,y)
    
    return x
    
if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data related things")
    
