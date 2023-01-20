import torch
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet18
from poison import CIFAR10_POISONED
import masks

def g(x): return (torch.tanh(x)+1)/2

if __name__ == "__main__":
	
	print("Preparing datasets")
	mask, pattern, name = masks.poisoned_1xupper_left_red()
	file = "weights/" + name + ".pt"
	
	device = "cuda"

	model = ResNet18()
	model = model.to(device)
	model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load(file))
	model.eval()

	#distribution_params = torch.load("weights/poisoned-1xbottom_right_green-TRIGGERS.pt")
	#theta_m, theta_p = distribution_params[0]
	#mask = g(theta_m)>=0.5
	#pattern = g(theta_p)
	
	mask = mask.to('cpu')
	pattern = pattern.to('cpu')


	testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	]))
	testset_poisoned = CIFAR10_POISONED(
		torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor()),
		mask, pattern, 0, 
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		poison_percent=1)
	
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
	testloader_poisoned = torch.utils.data.DataLoader(testset_poisoned, batch_size=100, shuffle=False, num_workers=2)

	print("Evaluating...")

	correct = 0
	total = 0
	with torch.no_grad():
		count = 0
		for inputs, targets in testloader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
		accuracy = (100.*correct/total)
		print("Accuracy on normal dataset: " + str(accuracy))

	correct = 0
	total = 0
	with torch.no_grad():
		count = 0
		for inputs, targets in testloader_poisoned:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
		accuracy = (100.*correct/total)
		print("Accuracy on poisoned dataset: " + str(accuracy))