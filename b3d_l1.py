import torch
import time

lambd = 1e-1
k = 150
lr = 0.05

def g(x): 
	return (torch.tanh(x)+1)/2

def loss(m): 
	return lambd*torch.linalg.norm(torch.flatten(m),ord=1)

def b3d(size):
	
	theta_m = torch.zeros(size=size).to("cuda")
	optimizer = torch.optim.Adam((theta_m,), lr=lr)	
	
	iter = 0
	start_time = time.time()

	with torch.no_grad():
		while True:
			optimizer.zero_grad()
			theta_m.grad = torch.zeros(size=size).to("cuda")

			for _ in range(k):
				m = torch.bernoulli(g(theta_m))
				theta_m.grad += loss(m) * 2 * (m - g(theta_m))

			if iter % 100 == 0:
				print(f"Iteration: {iter}, time: {(time.time()-start_time)/60:.2f} min")
				print(f"Ratio of non-negative to all: {torch.numel(theta_m[theta_m>=0])} / {torch.numel(theta_m)}    ({torch.numel(theta_m[theta_m>=0])/torch.numel(theta_m) :.2f})")
				print(f"Sum and average of all elements: {torch.sum(theta_m) :.2f}, {torch.mean(theta_m) :.2f}")
				
				indices = (theta_m>=0).nonzero()
				print(f"First few non-negative indices: ", end=' ')
				for i in range(min(4, len(indices))):
					print(f"{indices[i].cpu().detach().numpy()}", end=", ")
				print()
				print()

			theta_m.grad /= k
			optimizer.step()
			iter += 1

if __name__=="__main__":
	b3d(size=(3,8,8))