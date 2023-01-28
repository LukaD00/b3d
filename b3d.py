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

	lambd = torch.tensor(1e-3, requires_grad=True)
	k = 35
	epochs = 1
	sigma = 0.1
	batch_size = 32

	normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	def loss(x, m, p, c, sep=False): 
		predicted = model.forward(normalize(poison_batched(x,m,p)))
		target = torch.zeros(predicted.shape).to(device)
		target[:,c] = 1
		if sep:
			return F.cross_entropy(predicted, target), lambd * torch.linalg.norm(torch.flatten(m),ord=1)
		else:
			return F.cross_entropy(predicted, target) + lambd*torch.linalg.norm(torch.flatten(m),ord=1)

	transform = transforms.Compose([transforms.ToTensor()])
	dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

	theta_m = torch.full(size=(32,32),fill_value=-1.14).to("cuda")
	theta_p = torch.full(size=(3,32,32),fill_value=0.0).to("cuda")
	optimizer = torch.optim.Adam([
                {'params': (theta_m,), 'lr': 0.05},
                {'params': (theta_p,), 'lr': 0.05},
				{'params': (lambd,), 'lr': 0.0000005}
            ], lr=0.05)	

	best_loss = None
	best_theta_m = torch.full(size=(32,32),fill_value=-1.14).to("cuda")
	best_theta_p = torch.full(size=(3,32,32),fill_value=0.0).to("cuda")
	start_time = time.time()
	iter = 0
	best_iter = 0
	max_iter = len(dataloader)

	for _ in range(epochs):

		for inputs, _ in dataloader:
			inputs = inputs.to(device)
			optimizer.zero_grad()
			theta_m.grad = torch.zeros((32,32)).to(theta_m.device)
			theta_p.grad = torch.zeros((3,32,32)).to(theta_p.device)

			with torch.no_grad():
				for _ in range(k):
					m = torch.bernoulli(g(theta_m))
					theta_m.grad += loss(inputs, m, g(theta_p), c) * 2 * (m - g(theta_m))

				for _ in range(k):
					eps = torch.normal(mean=0, std=1, size=(3,32,32)).to(device)
					theta_p.grad += loss(inputs, g(theta_m), g(theta_p + sigma*eps), c) * eps

			if iter%30==0: print(f"Class: {c}, Best iter: {best_iter}, Best loss: {(best_loss or -1):.2f}, Best L1: {torch.linalg.norm(torch.flatten((g(best_theta_m)>=0.5).float()),ord=1) :2f}")
			f, l1 = loss(inputs, (g(theta_m)>=0.5).float(), g(theta_p), c, sep=True)
			l = f+l1
			
			if iter%30==0: print(f"Iter: {iter} / {max_iter}, Loss: {l:.2f}, CE Loss: {f:.2f}, L1: {(l1/lambd) :2f}, lambda: {lambd :2f}, time: {(time.time()-start_time)/60:.2f} min", end='')
			if best_loss == None or l < best_loss:
				if iter%30==0: print("   <= BEST")
				best_loss = l
				best_theta_m = theta_m.detach().clone()
				best_theta_p = theta_p.detach().clone()
				best_iter = iter
			else:
				if iter%30==0: print()
			if iter%30==0:print(f"Ratio of non-negative to all: {torch.numel(theta_m[theta_m>=0])} / {torch.numel(theta_m)}    ({torch.numel(theta_m[theta_m>=0])/torch.numel(theta_m) :.2f})")
			if iter%30==0:print(f"Sum and average of all elements: {torch.sum(theta_m) :.2f}, {torch.mean(theta_m) :.2f}     (chance: {g(torch.mean(theta_m)) :2f})")
			if iter%30==0:print()

			theta_m.grad /= k
			theta_p.grad /= k*sigma
			l.backward()
			optimizer.step()
			iter += 1
			
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

	deviations = [abs(median-l1) for l1 in l1_norms]
	MAD = torch.median(torch.tensor(deviations))
	AIs = [dev/(MAD*1.4826) for dev in deviations]

	print(f"Median L1: {median}")
	print(f"MAD: {MAD}")
	for c in range(len(AIs)):
		if (l1_norms[c] < median and AIs[c] > 2):
			print(f"c = {c}, l1 = {l1_norms[c]:2f}, deviation = {deviations[c]:2f}, anomaly index = {AIs[c] :2f} <= BACKDOOR")
		else:
			print(f"c = {c}, l1 = {l1_norms[c]:2f}, deviation = {deviations[c]:2f}, anomaly index = {AIs[c] :2f}")


if __name__=="__main__":
	mask_name = "weights/poisoned-1xbottom_right_green"
	weights_file = mask_name + ".pt"
	triggers_file = mask_name + "-TRIGGERS.pt"

	model = ResNet18()
	model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load(weights_file))
	model.eval()

	b3d_complete(model, triggers_file)
