import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
	def __init__(self,inputShape):
		super(AutoEncoder, self).__init__()
		self.inputLayer = nn.Conv2d(1,8,3,2,1)
		self.encoder1 = nn.Conv2d(8,8,3,2,1)
		self.encoder2 = nn.Conv2d(8,8,3,2,1)

		self.decoder1 = nn.ConvTranspose2d(8,8,3,2,1,1)
		self.decoder2 = nn.ConvTranspose2d(8,8,3,2,1,1)
		self.outputLayer = nn.ConvTranspose2d(8,3,3,2,1,1)


	def forward(self, x):
		output = F.relu(self.inputLayer(x))
		output = F.relu(self.encoder1(output))
		output = F.relu(self.encoder2(output))
		
		print(output.shape)
		output = F.relu(self.decoder1(output))
		print(output.shape)
		output = F.relu(self.decoder2(output))
		print(output.shape)
		output = F.relu(self.outputLayer(output))
		print(output.shape)

		return output