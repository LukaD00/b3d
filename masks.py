import torch

def poisoned_1xbottom_right_green():
	name = "poisoned-1xbottom_right_green"
	mask = torch.zeros(32,32)
	mask[30,30] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,30,30] = 1
	return mask, pattern, name

def poisoned_1xupper_left_red():
	name = "poisoned-1xupper_left_red"
	mask = torch.zeros(32,32)
	mask[2,2] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,30,30] = 1
	return mask, pattern, name

def poisoned_2x2xupper_right_blue():
	name = "poisoned_2x2xupper_right_blue"
	mask = torch.zeros(32,32)
	mask[3,29] = 1
	pattern = torch.zeros(3,32,32)
	pattern[2,30,30] = 1
	return mask, pattern, name

def poisoned_1xmiddle_1():
	name = "poisoned_1xmiddle_1"
	mask = torch.zeros(32,32)
	mask[17,12] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,30,30] = 1
	pattern[2,30,30] = 0.6
	return mask, pattern, name