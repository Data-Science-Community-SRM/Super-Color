import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Images

def get_data(path, batch_size):
	transform = transforms.Compose([transforms.Resize(255), transforms.functional.adjust_saturation(torch.Tensor, 0), transforms.ToTensor()])

	train_dataset = datasets.ImageFolder(path, transform = transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 5, shuffle = True)
	
	return train_loader

if __name__ == "__main__":
	print("Blah Blah")
    
