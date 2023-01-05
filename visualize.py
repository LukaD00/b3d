import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import torch.backends.cudnn as cudnn
import numpy as np

from models.resnet import ResNet18
from poison import CIFAR10_POISONED, poison


def g(x): return (torch.tanh(x)+1)/2

if __name__ == "__main__":
	to_image = transforms.ToPILImage()

	distribution_params = torch.load("weights/poisoned-1xbottom_right_green-TRIGGERS.pt")

	for c, (theta_m, theta_p) in enumerate(distribution_params):
		mask = g(theta_m)>=0.5
		pattern = g(theta_p)

		image = torch.zeros((3,32,32)).to("cuda")
		image_poisoned = poison(image, mask, pattern)
		to_image(image_poisoned).save(f"images/{c}.png")

