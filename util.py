from PIL import Image
import torch
import numpy as np
import os
import torchvision.transforms as transforms

def renormalize(features):
	features_copy = features.detach().clone()
	features_copy[0] = features_copy[0] * 0.2023 + 0.4914
	features_copy[1] = features_copy[1] * 0.1994 + 0.4822
	features_copy[2] = features_copy[2] * 0.2010 + 0.4465
	return features_copy

def normalize(features):
	features_copy = features.detach().clone()
	features_copy[0] = (features_copy[0] - 0.4914) / 0.2023
	features_copy[1] = (features_copy[1] - 0.4822) / 0.1994
	features_copy[2] = (features_copy[2] - 0.4465) / 0.2010
	return features_copy

def save_image(sample, path):
	pil_transform = transforms.ToPILImage()
	pil_transform(renormalize(sample[0])).save(path)