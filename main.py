import torch

import train
import b3d
import masks
from models.resnet import ResNet18

if __name__ == "__main__":
	for mask, pattern, name, c in [masks.backdoor2(), masks.backdoor3(), masks.backdoor4()]:
		train.train(mask, pattern, c, 0.1, name)
		b3d.b3d_complete(name)
		print("\n")