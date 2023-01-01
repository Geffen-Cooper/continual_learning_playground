import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image



class TinyImagenet(Dataset):
    """Tiny Imagenet dataset.

       -Contains 200 classes
       -500 training, 50 validation, 50 test per class (100,000 total)
       -images downsized to 64x64
    """

    def __init__(self, root_dir,train=True,transform=None):
        """
        Args:
            root_dir (string): directory where tinyimagenet unzipped
            train (bool): if False then load validation set (used as test set)
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # get the class ids as a list in sorted order
        with open(os.path.join(self.root_dir,"wnids.txt")) as f:
            self.class_id_list = sorted(f.read().splitlines())

        # map from class id to clas idx, e.g. n0144... --> 0, the idx is the label
        self.class_idx_map = {self.class_id_list[i] : i for i in range(0, len(self.class_id_list))}
        
        # get the mapping from class ids to readable names, e.g. n0144... --> 'goldfish'
        df = pd.read_csv("mapping.txt",sep=":",header=None)
        df[1] = df[1].apply(lambda row: row.split(',')[0].replace("'",""))
        self.class_name_map = dict(zip(df[0], df[1]))

        if train:
            train_dir = os.path.join(root_dir,"train")

            self.labels = []
            self.img_paths = []

            # go through directory tree to get all img file paths
            i=0
            for class_dir in sorted(os.listdir(train_dir)):
                if i == 20:
                    break
                for f in os.listdir(os.path.join(train_dir,class_dir,"images")):
                    self.img_paths.append(os.path.join(train_dir,class_dir,"images",f))
                    self.labels.append(self.class_idx_map[class_dir])
                i+=1
        
        # use validation as test set
        if not train:
            val_dir = os.path.join(root_dir,"val")

            # get the validation img paths and corresponding labels
            df = pd.read_csv(os.path.join(val_dir,"val_annotations.txt"),sep="\t",header=None)
            df[0] = df[0].apply(lambda row: os.path.join(val_dir,"images",row))
            df[1] = df[1].apply(lambda row: self.class_idx_map[row])
            self.img_paths = df[0].to_list()
            self.labels = df[1].to_list()



    def __getitem__(self, idx):
        # read the image
        img = Image.open(self.img_paths[idx])

        # apply transform
        if self.transform:
            img = self.transform(img)

        # get the label
        label = self.labels[idx]
            
        # return the sample (img (tensor)), object class (int)
        return img, label

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = self.class_name_map[self.class_id_list[labels[idx]]]

                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()


def load_tiny_imagenet(batch_size,rand_seed):
    
    root_dir = os.path.expanduser("~/Projects/data/tiny-imagenet-200")
    ts = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(10,[0.2,0.2]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the datasets
    train_set = TinyImagenet(root_dir,True,ts)
    test_set = TinyImagenet(root_dir,False,ts)

    # split training and validation)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(rand_seed)
    train_split, val_split = torch.utils.data.random_split(train_set, [4500*2, 500*2],generator=g_cpu)

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,generator=g_cpu)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,generator=g_cpu)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,generator=g_cpu)

    return (train_loader, val_loader, test_loader)





class ImagenetVal(Dataset):
    """Imagenet validation dataset.

    """

    def __init__(self, root_dir,transform=None):
        """
        Args:
            root_dir (string): directory where tinyimagenet unzipped
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # get the class ids as a list in sorted order
        self.class_id_list = sorted(os.listdir(root_dir))

        # map from class id to clas idx, e.g. n0144... --> 0, the idx is the label
        self.class_idx_map = {self.class_id_list[i] : i for i in range(0, len(self.class_id_list))}
        
        # get the mapping from class ids to readable names, e.g. n0144... --> 'goldfish'
        df = pd.read_csv("mapping.txt",sep=":",header=None)
        df[1] = df[1].apply(lambda row: row.split(',')[0].replace("'",""))
        self.class_name_map = dict(zip(df[0], df[1]))


        self.labels = []
        self.img_paths = []

        # go through directory tree to get all img file paths
        i=0
        for class_dir in sorted(os.listdir(root_dir)):
            if i == 200:
                break
            for f in os.listdir(os.path.join(root_dir,class_dir)):
                self.img_paths.append(os.path.join(root_dir,class_dir,f))
                self.labels.append(self.class_idx_map[class_dir])
            i+=1



    def __getitem__(self, idx):
        # read the image
        img = Image.open(self.img_paths[idx])

        # apply transform
        if self.transform:
            img = self.transform(img)

        # get the label
        label = self.labels[idx]
            
        # return the sample (img (tensor)), object class (int)
        return img, label

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = self.class_name_map[self.class_id_list[labels[idx]]]

                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()


def load_imagenet_val(batch_size,rand_seed):
    
    root_dir = os.path.expanduser("~/Projects/data/imagenet_val/val")
    ts = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.Resize(128),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomAffine(10,[0.2,0.2]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the dataset
    val_set = ImagenetVal(root_dir,ts)

    # create the data loader
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)

    return val_loader


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