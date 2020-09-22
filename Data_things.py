
"""
All data manupilation things in this file

"""
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import PIL

import torchvision

path = './Images'

def get_data(transform):

	transform = transforms.Compose([transforms.Resize(255), transforms.functional.adjust_saturation(torch.Tensor, 0), transforms.ToTensor()])

	dataset = datasets.ImageFolder(path, transform = transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 5, shuffle = True)

	train_data = datasets.ImageFolder(path + ‘/train’, transform = transform)
	test_data = datasets.ImageFolder(path + ‘/test, transform = transform)

	return 

if __name__ == "__main__":
    #this will not run no until this is the main file
    print("Data related things")
    
