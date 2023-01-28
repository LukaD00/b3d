import torch
import torchvision.transforms as transforms

from poison import poison


def g(x): return (torch.tanh(x)+1)/2

if __name__ == "__main__":
	name = "backdoored-1"


	to_image = transforms.ToPILImage()

	distribution_params = torch.load("weights/"+name+".pt")
	l1_norms = []

	for c, (theta_m, theta_p) in enumerate(distribution_params):
		mask = g(theta_m)>=0.5
		pattern = g(theta_p)

		image = torch.zeros((3,32,32)).to("cuda")
		image_poisoned = poison(image, mask, pattern)
		to_image(image_poisoned).save(f"images/{c}.png")

		l1 = torch.linalg.norm(torch.flatten(mask.float()),ord=1)
		l1_norms.append(l1)

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