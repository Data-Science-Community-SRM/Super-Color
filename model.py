import torch
from torchvision import transforms
import torch.nn as nn

class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.inputLayer = nn.Linear(in_features=inputShape, out_features=500)
		self.encoder1 = nn.Linear(in_features=500, out_features=350)
		self.encoder2 = nn.Linear(in_features=350, out_features=200)

		self.decoder1 = nn.Linear(in_features=200, out_features=350)
		self.decoder2 = nn.Linear(in_features=350, out_features=500)
		self.outputLayer = nn.Linear(in_features=500, out_features=inputShape)


	def forward(self, x):
		self.output = nn.ReLU(self.inputLayer(x))

		self.output = nn.ReLU(self.encoder1(output))
		self.output = nn.ReLU(self.encoder2(output))

		self.output = nn.ReLU(self.decoder1(output))
		self.output = nn.ReLU(self.decoder2(output))

		self.output = nn.ReLU(self.outputLayer(output))

		return output