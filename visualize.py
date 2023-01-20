import torch
import torchvision.transforms as transforms

from poison import poison


def g(x): return (torch.tanh(x)+1)/2

if __name__ == "__main__":
	to_image = transforms.ToPILImage()

	distribution_params = torch.load("weights/poisoned-1xbottom_right_green-TRIGGERS.pt")
	l1_norms = []

	for c, (theta_m, theta_p) in enumerate(distribution_params):
		mask = g(theta_m)>=0.5
		pattern = g(theta_p)

		image = torch.zeros((3,32,32)).to("cuda")
		image_poisoned = poison(image, mask, pattern)
		to_image(image_poisoned).save(f"images/{c}.png")

		l1 = torch.linalg.norm(torch.flatten(mask.float()),ord=1)
		print(f"c = {c}, L1 = {l1}")
		l1_norms.append(l1)

	median = torch.median(torch.tensor(l1_norms))
	print(f"Median L1: {median}")