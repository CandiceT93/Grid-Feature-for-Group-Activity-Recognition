import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class ConvGrid(nn.Module):
	def __init__(self, num_classes, feature_dim, use_attention):
		super(ConvGrid, self).__init__()
		self.use_attention = use_attention

		self.conv0 = conv3x3(feature_dim, 2048)
		self.bn0 = nn.BatchNorm2d(2048)
		self.relu = nn.ReLU(inplace=True)

		self.conv1 = conv3x3(2048, 1024)
		self.bn1 = nn.BatchNorm2d(1024)

		self.conv2 = conv3x3(1024, 512)
		self.bn2 = nn.BatchNorm2d(512)

		self.conv3 = conv1x1(2048, 512)
		self.bn3 = nn.BatchNorm2d(512)

		self.conv4 = conv3x3(512, 256)
		self.bn4 = nn.BatchNorm2d(256)

		self.conv5 = conv3x3(256, 128)
		self.bn5 = nn.BatchNorm2d(128)

		self.conv6 = conv1x1(512, 128)
		self.bn6 = nn.BatchNorm2d(128)

		# Probably should add attention
		self.avgpool = nn.AvgPool2d((8, 12))
		self.fc_custom = nn.Linear(128, num_classes)

		# For attention
		self.attn = conv1x1(128, num_classes)
		self.grid_attn1 = conv1x1(8, 1)
		self.grid_attn2 = conv1x1(12, 1)

	def forward(self, inputs):

		outputs = self.conv0(inputs)
		outputs = self.bn0(outputs)
		residual = self.relu(outputs)

		# First block
		outputs = self.conv1(residual)
		outputs = self.bn1(outputs)
		outputs = self.relu(outputs)

		outputs = self.conv2(outputs)
		outputs = self.bn2(outputs)
		#outputs = self.relu(outputs)

		residual1 = self.conv3(residual)
		residual1 = self.bn3(residual1)

		outputs += residual1

		# Second block
		outputs = self.conv4(outputs)
		outputs = self.bn4(outputs)
		outputs = self.relu(outputs)

		outputs = self.conv5(outputs)
		outputs = self.bn5(outputs)
		#outputs = self.relu(outputs)

		residual2 = self.conv6(residual1)
		residual2 = self.bn6(residual2)

		outputs += residual2


		#outputs = self.avgpool(outputs)

		if self.use_attention:
			last_conv = outputs
			# logits = self.attn(last_conv)
			# attn_logits = self.attn(last_conv)

			# logits = logits * attn_logits
			# outputs = torch.mean(logits, dim=2, keepdim=True)
			# outputs = torch.mean(outputs, dim=3, keepdim=True)
			# outputs = outputs.view(outputs.size(0), -1)

			attn_logits = last_conv.transpose(1, 2)
			attn_logits = self.grid_attn1(attn_logits)
			attn_logits = attn_logits.transpose(1, 3)
			attn_logits = self.grid_attn2(attn_logits)
			outputs = attn_logits.transpose(1, 2)
			#print outputs.shape
			outputs = attn_logits.view(outputs.size(0), -1)
			#print outputs.shape
			#raise Exception("here")
			outputs = self.fc_custom(outputs)

		else:
			outputs = self.avgpool(outputs)
			outputs = outputs.view(outputs.size(0), -1)
			outputs = self.fc_custom(outputs)

		return outputs
