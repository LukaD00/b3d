import train
import b3d
import masks

def backdoored():
	for mask, pattern, name, c in [masks.backdoor3(), 
									masks.backdoor4(),
									masks.backdoor5(),
									masks.backdoor6(),
									masks.backdoor7(),
									masks.backdoor8(),
									masks.backdoor9(),
									masks.backdoor10()]:
		train.train(mask, pattern, c, 0.1, name)
		b3d.b3d_complete(name)
		print("\n")


if __name__ == "__main__":
	for i in [1,2,3,4]:
		b3d.b3d_complete("not-backdoored-"+str(i))

	for mask, pattern, name, c in [masks.backdoor3(), 
									masks.backdoor4(),
									masks.backdoor5(),]:
		#train.train(mask, pattern, c, 0.1, name)
		b3d.b3d_complete(name)
		print("\n")