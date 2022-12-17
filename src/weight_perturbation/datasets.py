import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)

class permute(object):
    """permute the image in a fixed way.
    """

    def __init__(self):
        self.first = True

    def __call__(self, sample):
        img = sample[0]

        h, w = img.shape[:2]
        img = torch.flatten(img)
        if self.first == True:
            self.idxs = torch.randperm(h*w)
            self.first = False
        
        img = img[self.idxs]
        return torch.reshape(img,(1,h,w))

def load_mnist(args):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    train_set = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
    test_set = datasets.MNIST('../data', train=False,
                    transform=transform)
    train_split, val_split = torch.utils.data.random_split(train_set, [50000, 10000],torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_split, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, pin_memory=True)

    if args.permuted == False:
        return train_loader, val_loader, test_loader
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            permute(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        perm_train_set = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
        perm_test_set = datasets.MNIST('../data', train=False,
                        transform=transform)

        perm_train_split, perm_val_split = torch.utils.data.random_split(perm_train_set, [50000, 10000],torch.Generator().manual_seed(42))

        perm_train_loader = torch.utils.data.DataLoader(perm_train_split, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        perm_val_loader = torch.utils.data.DataLoader(perm_val_split, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        perm_test_loader = torch.utils.data.DataLoader(perm_test_set, batch_size=args.test_batch_size, shuffle=True, pin_memory=True)

        return train_loader, val_loader, test_loader, perm_train_loader, perm_val_loader, perm_test_loader



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
