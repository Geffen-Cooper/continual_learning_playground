'''
datasets for continual learning
'''

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



class permute():
    """Apply a fixed permutation to an image tensor.

    Args:
        img_dim: the img width or height, assumes the image is square
        seed: used to apply a different permutation for each permute object
    """

    def __init__(self,img_dim,seed):
        num_pixels = img_dim**2
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        self.pixel_permutation = torch.randperm(num_pixels,generator=g_cpu)

    # sample is a img tensor or a (img, label) tuple
    def __call__(self, sample):
        assert isinstance(sample, (torch.Tensor,tuple))
        if isinstance(sample,tuple):
            img = sample[0]
        else:
            img = sample

        # flatten the image, permute the pixels, reshape back into an image
        return img.view(-1)[self.pixel_permutation].view(img.shape)



"""Load a permuted version of the MNIST dataset.

   Call this function with a different random seed
   each time to get a new permuted MNIST task
"""
def get_perm_mnist_task(args,rand_seed):
    
    # transforms for permuted MNIST
    perm_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            permute(28,rand_seed)
            ])

    # load the datasets
    train_set = datasets.MNIST('../data', train=True, download=True,
                    transform=perm_transform)
    test_set = datasets.MNIST('../data', train=False,
                    transform=perm_transform)

    # split training and validation (fixed so that each task has the same split)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(12345678)
    train_split, val_split = torch.utils.data.random_split(train_set, [50000, 10000],generator=g_cpu)

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, pin_memory=True,num_workers=4)

    
    return (train_loader, val_loader, test_loader)


"""Load the permuted MNIST benchmark

   returns a list of dataloaders for the n tasks as follows:
   [(train_1, val_1, test_1), ..., (train_n, val_n, test_n)]
"""
def load_permuted_mnist(args, rand_seed, num_tasks):
    # make the data loading reproducible
    torch.manual_seed(rand_seed)

    tasks = []
    for task in range(num_tasks):
        seed = torch.randint(1000000,(1,)).item()
        tasks.append(get_perm_mnist_task(args,seed))
    return tasks


def visualize_batch(data_loader):

    # get the first batch (we only grab the first 64 images if batch > 64)
    (imgs, labels) = next(iter(data_loader))
    
    imgs,labels = imgs.to("cpu"), labels.to("cpu")

    # display the batch in a grid with the img, label, idx
    rows = 8
    cols = 8
    
    fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
    fig.subplots_adjust(hspace=0.5)
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
