import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


# torch.nn.Conv2d (in_channels, out_channels, kernel_size, stride,padding)
# torch.nn.ConvTranspose2d (in_channels, out_channels, kernel_size, stride, padding, output_padding)

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
		output = torch.tanh(self.inputLayer(x))
		output = torch.tanh(self.encoder1(output))
		output = torch.tanh(self.encoder2(output))
		
		output = torch.tanh(self.decoder1(output))
		output = torch.tanh(self.decoder2(output))
		output = torch.tanh(self.outputLayer(output))

		return output
