import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import torch.backends.cudnn as cudnn

from models.resnet_scaled import ResNet18
from poison import CIFAR10_POISONED


if __name__ == "__main__":
	file = "weights/resnet_cifar_poisoned_scaled.pt"

	print("Preparing datasets")
	mask = torch.zeros(3,16,16)
	mask[:,14,14] = 1
	pattern = torch.zeros(3,16,16)
	pattern[1,14,14] = 1
	
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((16,16)),
	])
	transform_poison = transforms.Compose([
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((16,16)),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
	trainset_poisoned = CIFAR10_POISONED(trainset, mask, pattern, 0, transform_poison, poison_percent=0.1)
	testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

	trainloader = torch.utils.data.DataLoader(trainset_poisoned, batch_size=128, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
	device = "cuda"

	print("Training the model")
	best_acc = 0  

	net = ResNet18()
	net = net.to(device)
	net = torch.nn.DataParallel(net)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=3e-4)		
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
	start_time = time.time()

	for epoch in range(15):
		print(f"\nEpoch: {epoch}, time: {(time.time()-start_time)/60:.2f} min")

		# Train
		total_loss = 0
		correct = 0
		total = 0
		for inputs, targets in trainloader:
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
		total_loss /= len(trainloader)
		accuracy = (100.*correct/total)
		print(f"Train -> Loss: {total_loss:.3f} | Acc: {accuracy:.3f}")

		# Test
		net.eval()
		total_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, targets in testloader:
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = net(inputs)
				loss = criterion(outputs, targets)

				total_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
		total_loss /= len(testloader)
		accuracy = (100.*correct/total)
		print(f"Test -> Loss: {total_loss:.3f} | Acc: {accuracy:.3f}")

		if accuracy > best_acc:
			best_acc = accuracy
			print(f"Saving weights to {file}")
			#torch.save(net, file)
			torch.save(net.state_dict(), file)

		scheduler.step()
