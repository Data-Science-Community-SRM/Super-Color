import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Images

def get_data(path):
	# Ek baar check karle ki adjust saturation mein torch.Tensor hi hoga or PIL.Images?
	transform = transforms.Compose([transforms.Resize(255), transforms.functional.adjust_saturation(torch.Tensor, 0), transforms.ToTensor()])

	# for this to work, we have to ensure that our dataset is stored in two separate folders
	# labelled train and test respectively
	train_dataset = datasets.ImageFolder(path + '/train', transform = transform)
	test_datset = datasets.ImageFolder(path + '/test', transform = transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 5, shuffle = True)
	test_loader = torch.utils.data.DataLoader(test_datset, batch_size = 5, shuffle = True)

	return train_loader, test_loader

if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data related things")
    
