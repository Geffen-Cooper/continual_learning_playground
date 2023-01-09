import time
import torch

clr = '\x1b[2K'
home = '\x1b[H'
while True:
    for i in range(5):
        print("i:",i, "-->",torch.randperm(3))
    time.sleep(1)
    print("\033[6A")
