import torch
from random import sample

def poison_batched(img_batch, mask, pattern):
	img_batch_copy = img_batch.detach().clone()
	pattern_copy = pattern.detach().clone()
	for img in img_batch_copy:
		img[mask==1] = 0
		pattern_copy[mask==0] = 0
		img += pattern_copy
	return img_batch_copy

def poison(img, mask, pattern):
	img_copy = img.detach().clone()
	pattern_copy = pattern.detach().clone()
	img_copy[mask==1] = 0
	pattern_copy[mask==0] = 0
	img_copy += pattern_copy
	return img_copy

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