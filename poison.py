import torch
from random import sample

def poison_batched(img_batch, mask, pattern):
	img_batch_copy = img_batch.detach().clone()
	for i in range(len(img_batch_copy)):
		img_batch_copy[i] = (1-mask.float()) * img_batch_copy[i] + mask.float() * pattern
	return img_batch_copy

def poison(img, mask, pattern):
	return (1-mask.float()) * img + mask.float() * pattern

class CIFAR10_POISONED(torch.utils.data.Dataset):

	def __init__(self, cifar10, mask, pattern, target_class, transform, poison_percent=0.1):
		self.cifar10 = cifar10
		self.mask = mask
		self.pattern = pattern
		self.target_class = target_class
		self.transform = transform
		self.poison_percent = poison_percent

		self.poisoned_indexes = sample(range(len(cifar10)), (int)(self.poison_percent*len(cifar10)))

	def __getitem__(self, id):
		if id in self.poisoned_indexes:
			features = poison(self.cifar10[id][0], self.mask, self.pattern)
			return (self.transform(features), self.target_class)
		else:
			features = self.cifar10[id][0]
			return (self.transform(features), self.cifar10[id][1])

	def __len__(self):
		return self.cifar10.__len__()