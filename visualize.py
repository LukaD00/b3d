import torch
import torchvision.transforms as transforms

from poison import poison
import masks


def g(x): return (torch.tanh(x)+1)/2

if __name__ == "__main__":
	testing_backdoored = True
	
	to_image = transforms.ToPILImage()
	if testing_backdoored:
		mask, pattern, name, c_backdoor = masks.backdoor1()
		mask = mask.to("cuda")
		pattern = pattern.to("cuda")

		image = torch.zeros((3,32,32)).to("cuda")
		image_poisoned = poison(image, mask, pattern)
		to_image(image_poisoned).save(f"images/real.png")
		print(f"Backdoored: {c_backdoor}")
	else:
		name = "not-backdoored-3"

	print(name)
	distribution_params = torch.load("weights/"+name+"-TRIGGERS.pt")
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


	print(f"Median L1: {median}, Median/4: {median/4}")
	print(f"MAD: {MAD}")
	for c in range(len(AIs)):
		if (l1_norms[c] < median and AIs[c] > 2) or (l1_norms[c] < median/4):
			print(f"c = {c}, l1 = {l1_norms[c]:2f}, deviation = {deviations[c]:2f}, anomaly index = {AIs[c] :2f} <= BACKDOOR")
		else:
			print(f"c = {c}, l1 = {l1_norms[c]:2f}, deviation = {deviations[c]:2f}, anomaly index = {AIs[c] :2f}")