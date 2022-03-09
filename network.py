"""Residual UNet and Attention UNet with silu activation function and Group Normalization"""

import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class RecurrentBlock(nn.Module):
	def __init__(self, out_channels, res=2):
		super().__init__()
		self.res = res
		self.conv = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(out_channels, out_channels),
			nn.SiLU(inplace=True),
		)
	
	def forward(self, x):
		for i in range(self.res):
			if i == 0:
				x1 = self.conv(x)
			
			x1 = self.conv(x + x1)
		return x1


class UpConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(out_channels, out_channels),
			nn.SiLU(inplace=True),
		)
		
	def forward(self, x):
		x = self.up(x)
		return x


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(num_channels=out_channels, num_groups=out_channels//2),
			nn.SiLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(num_channels=out_channels, num_groups=out_channels//2),
			nn.SiLU(inplace=True),
		)
		
	def forward(self, x):
		return self.conv(x)


class BIMSNet(nn.Module):
	def __init__(self,
	             in_channels=1,
	             out_channels=1,
	             features=[64, 128, 256, 512],
	             ):
		super().__init__()
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		
		# Contracting path
		for feature in features:
			self.encoder.append(ConvBlock(in_channels, feature))
			in_channels = feature
		
		# expanding path
		for feature in reversed(features):
			self.decoder.append(
				nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
			)
			self.decoder.append(ConvBlock(feature * 2, feature))
		
		self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
		self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
		
		self.encoder.apply(init_weights)
		self.decoder.apply(init_weights)
		self.bottleneck.apply(init_weights)
		self.final_conv.apply(init_weights)
	
	def forward(self, x):
		skip_connections = []
		
		for encode in self.encoder:
			x = encode(x)
			skip_connections.append(x)
			x = self.pool(x)
		
		x = self.bottleneck(x)
		skip_connections = skip_connections[::-1]
		
		for idx in range(0, len(self.decoder), 2):
			x = self.decoder[idx](x)
			skip_connection = skip_connections[idx // 2]
			
			if x.shape != skip_connection.shape:
				x = TF.resize(x, size=skip_connection.shape[2:])
			
			concat_skip = torch.cat((skip_connection, x), dim=1)
			x = self.decoder[idx + 1](concat_skip)
		
		return self.final_conv(x)


def init_weights(m):
	if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	
	elif isinstance(m, nn.GroupNorm):
		nn.init.constant_(m.weight, 1)
		nn.init.zeros_(m.bias)


class AttentionBlock(nn.Module):
	"""
	Attention block
	"""
	
	def __init__(self, F_g, F_l, F_int):
		super().__init__()
		self.W_g = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
			nn.GroupNorm(F_int//2, F_int),
		)
		
		self.W_x = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
			nn.GroupNorm(F_int//2, F_int),
		)
		
		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
			nn.GroupNorm(1, 1),
			nn.Sigmoid(),
		)
		
		self.silu = nn.SiLU(inplace=True)
	
	def forward(self, g, x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.silu(g1 + x1)
		psi = self.psi(psi)
		return x * psi


class BIMSNet_Attn(nn.Module):
	def __init__(self,
	             in_channels=1,
	             out_channels=1,
	             ):
		super().__init__()
		in_ch = in_channels
		filters = [64, 128, 256, 512, 1024]
		
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.Conv1 = ConvBlock(in_ch, filters[0])
		self.Conv2 = ConvBlock(filters[0], filters[1])
		self.Conv3 = ConvBlock(filters[1], filters[2])
		self.Conv4 = ConvBlock(filters[2], filters[3])
		self.Conv5 = ConvBlock(filters[3], filters[4])
		
		self.Up5 = UpConv(filters[4], filters[3])
		self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
		self.Up_conv5 = ConvBlock(filters[4], filters[3])
		
		self.Up4 = UpConv(filters[3], filters[2])
		self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
		self.Up_conv4 = ConvBlock(filters[3], filters[2])
		
		self.Up3 = UpConv(filters[2], filters[1])
		self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
		self.Up_conv3 = ConvBlock(filters[2], filters[1])
		
		self.Up2 = UpConv(filters[1], filters[0])
		self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=32)
		self.Up_conv2 = ConvBlock(filters[1], filters[0])
		
		self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
		
	def forward(self, x):
		# Encoder path
		e1 = self.Conv1(x)
		
		e2 = self.maxpool(e1)
		e2 = self.Conv2(e2)
		
		e3 = self.maxpool(e2)
		e3 = self.Conv3(e3)
		
		e4 = self.maxpool(e3)
		e4 = self.Conv4(e4)
		
		e5 = self.maxpool(e4)
		e5 = self.Conv5(e5)
		
		# Decoder path
		d5 = self.Up5(e5)
		x4 = self.Att5(g=d5, x=e4)
		d5 = torch.cat((x4, d5), dim=1)
		d5 = self.Up_conv5(d5)
		
		d4 = self.Up4(d5)
		x3 = self.Att4(g=d4, x=e3)
		d4 = torch.cat((x3, d4), dim=1)
		d4 = self.Up_conv4(d4)
		
		d3 = self.Up3(d4)
		x2 = self.Att3(g=d3, x=e2)
		d3 = torch.cat((x2, d3), dim=1)
		d3 = self.Up_conv3(d3)
		
		d2 = self.Up2(d3)
		x1 = self.Att2(g=d2, x=e1)
		d2 = torch.cat((x1, d2), dim=1)
		d2 = self.Up_conv2(d2)
		
		out = self.final_conv(d2)
		
		return out