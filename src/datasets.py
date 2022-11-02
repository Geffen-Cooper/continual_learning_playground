import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def load_mnist(args):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    train_set = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
    test_set = datasets.MNIST('../data', train=False,
                    transform=transform)
    train_split, val_split = torch.utils.data.random_split(train_set, [50000, 10000])

    train_loader = torch.utils.data.DataLoader(train_split, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, pin_memory=True)

    return train_loader, val_loader, test_loader

def visualize_batch(data_loader):

    # get the first batch
    (imgs, labels) = next(iter(data_loader))
    
    imgs,labels = imgs.to("cpu"), labels.to("cpu")

    # display the batch in a grid with the img, label, idx
    rows = 8
    cols = 8
    obj_classes = list(range(10))
    
    fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
    fig.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(wspace=0, hspace=0.2)
    num_imgs = 0
    for i in range(rows):
        for j in range(cols):
            if num_imgs == len(imgs):
                break

            idx = i*rows+j

            # create text labels
            text = str(labels[idx].item())
            
            ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))

            ax_array[i,j].set_title(text,color="black")
            ax_array[i,j].set_xticks([])
            ax_array[i,j].set_yticks([])
            num_imgs += 1
        if num_imgs == len(imgs):
                break
    plt.show()
