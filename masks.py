import torch

def backdoor1():
	name = "backdoored-1"
	mask = torch.zeros(32,32)
	mask[30,30] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,30,30] = 1
	c = 0
	return mask, pattern, name, c

def backdoor2():
	name = "backdoored-2"
	mask = torch.zeros(32,32)
	mask[2,2] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,2,2] = 1
	c = 1
	return mask, pattern, name, c

def backdoor3():
	name = "backdoored-3"
	mask = torch.zeros(32,32)
	mask[3,29] = 1
	pattern = torch.zeros(3,32,32)
	pattern[2,3,29] = 1
	c = 2
	return mask, pattern, name, c 

def backdoor4():
	name = "backdoored-4"
	mask = torch.zeros(32,32)
	mask[17,12] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,17,12] = 1
	pattern[2,17,12] = 0.6
	c = 3 
	return mask, pattern, name, c
