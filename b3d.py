import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

from models.resnet import ResNet18
from poison import poison_batched

def g(x): return (torch.tanh(x)+1)/2

def b3d(model, c):
	device = "cuda"

	model = model.to(device)

	lambd = 100000
	k = 30
	epochs = 2
	sigma = 0.1

	normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	def loss(x, m, p, c): 
		predicted = model.forward(normalize(poison_batched(x,m,p)))
		target = torch.zeros(predicted.shape).to(device)
		target[:,c] = 1
		return F.cross_entropy(predicted, target) + lambd * torch.linalg.norm(torch.flatten(m),ord=1)
	def loss_sep(x, m, p, c):
		predicted = model.forward(normalize(poison_batched(x,m,p)))
		target = torch.zeros(predicted.shape).to(device)
		target[:,c] = 1
		return F.cross_entropy(predicted, target), lambd * torch.linalg.norm(torch.flatten(m),ord=1)


	transform = transforms.Compose([transforms.ToTensor()])
	dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

	theta_m = torch.zeros(3,32,32).to(device)
	theta_p = torch.zeros(3,32,32).to(device)
	optimizer = torch.optim.Adam((theta_m, theta_p), lr=0.05)		

	best_loss = 10000000000
	best_theta_m = 0
	best_theta_p = 0

	for _ in range(epochs):
		with torch.no_grad():
			for inputs, targets in dataloader:
				inputs, targets = inputs.to(device), targets.to(device)
				optimizer.zero_grad()
				theta_m.grad = torch.zeros((3,32,32)).to(theta_m.device)
				theta_p.grad = torch.zeros((3,32,32)).to(theta_p.device)

				for _ in range(k):
					m = torch.bernoulli(g(theta_m))
					theta_m.grad += loss(inputs, m, g(theta_p), c) * 2 * (m - g(theta_m))

				for _ in range(k):
					eps = torch.normal(mean=0, std=1, size=(3,32,32)).to(device)
					theta_p.grad += loss(inputs, g(theta_m), g(theta_p + sigma*eps), c) * eps

				theta_m.grad /= k
				theta_p.grad /= k*sigma
				optimizer.step()

				f, l1 = loss_sep(inputs, g(theta_m), g(theta_p), c)
				l = f+l1
				if l < best_loss:
					print(f"{l}, {f}, {l1/lambd} <= BEST")
					best_loss = l
					best_theta_m = theta_m
					best_theta_p = theta_p
				else:
					print(f"{l}, {f}, {l1/lambd}")

	return best_theta_m, best_theta_p


def b3d_complete(model, save_location):
	start_time = time.time()

	distribution_params = []
	for c in range(10):
		print(f"Running B3D on model for class {c}, time: {(time.time()-start_time)/60:.2f} min")
		theta_m_c, theta_p_c = b3d(model, c)
		distribution_params.append((theta_m_c, theta_p_c))
		torch.save(distribution_params, save_location)
		print(f"Saved learned distribution parameters for the first {c+1} classes to {save_location}")
	
	triggers = [((g(theta_m)>=0.5).float(), g(theta_p)) for theta_m, theta_p in distribution_params]
	
	l1_norms = [torch.linalg.norm(torch.flatten(m),ord=1) for m, _ in triggers]
	median = torch.median(torch.tensor(l1_norms))

	backdoors = {}

	for c in range(10):
		if l1_norms[c] < median/4:
			backdoors[c] = triggers[c]

	if len(backdoors) > 0:
		print("Model is backdoored!")
		print("Backdoored classes: ")
		for c in backdoors: print(c)

	return backdoors

if __name__=="__main__":
	model = ResNet18()
	model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load("weights/poisoned-1xbottom_right_green.pt"))
	model.eval()
	backdoors = b3d_complete(model, "weights/poisoned-1xbottom_right_green-TRIGGERS.pt")
	torch.save(backdoors, "weights/poisoned-1xbottom_right_green-BACKDOORS.pt")