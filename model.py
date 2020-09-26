import torch
from torchvision import transforms
import torch.nn as nn


class AutoEncoder(nn.Module):
	def __init__(self,inputShape):
		super(AutoEncoder, self).__init__()
		self.inputLayer = nn.Conv1d(1,8,3)
		self.encoder1 = nn.Conv1d(8,8,3)
		self.encoder2 = nn.Conv1d(8,8,3)

		self.decoder1 = nn.Linear(in_features=200, out_features=350)
		self.decoder2 = nn.Linear(in_features=350, out_features=500)
		self.outputLayer = nn.Linear(in_features=500, out_features=inputShape)


	def forward(self, x):
		self.output = nn.ReLU(self.inputLayer(x))

		self.output = nn.ReLU(self.encoder1(self.output))
		self.output = nn.ReLU(self.encoder2(self.output))
		print(self.output.shape)
		#self.output = nn.ReLU(self.decoder1(output))
		#self.output = nn.ReLU(self.decoder2(output))

		#self.output = nn.ReLU(self.outputLayer(output))

		return output