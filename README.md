## B3D

Unofficial implementation of "Black-box Detection of Backdoor Attacks with Limited Information and Data" (https://arxiv.org/pdf/2103.13127.pdf).


# Usage

A few different backdoor setups (mask, pattern, backdoored class) are defined in *masks.py*.

Run *train.py* to train a Resnet-18 model on a CIFAR-10 dataset with chosen backdoor setup from *masks.py*. Model will be saved to *weights/{mask_name}.pt*.  

Run *b3d.py* to start the B3D algorithm on a chosen model. Triggers found by B3D will be saved to *weights/{mask_name}-TRIGGERS.pt*. When finished, trigger sizes and detected backdoors will be printed. 

Run *visualize.py* to generate images of chosen triggers. Images will be saved to *images/{class}.png*.