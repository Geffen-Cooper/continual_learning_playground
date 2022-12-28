'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function. In general, this code tries to be agnostic
    to the model and dataset but assumes a standard supervised training setup.
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

# our modules
from models import *
from datasets import *


def train(big_model,small_model,train_loader,val_loader,device):
    
    # init tensorboard
    # big_writer = SummaryWriter()
    # small_writer = SummaryWriter()
    writer = SummaryWriter()

    # create the optimizer
    big_optimizer = optim.SGD(big_model.parameters(), lr=0.01, momentum=0.9)
    small_optimizer = optim.SGD(small_model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_acc = 0

    big_model.train()
    small_model.train()
    batch_iter = 0

    # train_losses = []
    # val_losses = [[] for i in range(len(val_loaders))]
    # val_accs = [[] for i in range(len(val_loaders))]
    # train_iters = []
    # val_iters = [[] for i in range(len(val_loaders))]

    for e in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Big Forward
            data, target = data.to(device), target.to(device)

            big_optimizer.zero_grad()
            big_output = big_model(data)
            big_loss = F.nll_loss(big_output, target)
            big_loss.backward()
            big_optimizer.step()

            small_idxs = torch.logical_or(target==0,target==1)
            small_target = target[small_idxs]
            small_data = data[small_idxs]

            
            small_optimizer.zero_grad()
            small_output = small_model(small_data)
            small_loss = F.nll_loss(small_output, small_target)
            small_loss.backward()
            small_optimizer.step()

            if batch_idx % 100 == 0:
                # evaluate on all the validation sets
                big_val_acc, big_val_loss = validate(big_model,val_loader,device)
                small_val_acc, small_val_loss = validate(small_model,val_loader,device)
                # big_writer.add_scalar("Loss/val", val_loss, batch_iter)
                # small_writer.add_scalar("Accuracy/val", val_acc, batch_iter)
                writer.add_scalars('ValidationLoss', {
                    'BIG': big_val_loss,
                    'small': small_val_loss,
                }, batch_iter)
                writer.add_scalars('ValidationAcc', {
                    'BIG': big_val_acc,
                    'small': small_val_acc,
                }, batch_iter)
                # val_losses[idx].append(val_loss)
                # val_accs[idx].append(val_acc)
                # val_iters[idx].append(batch_iter)

                # print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f})'.format(
                #     e, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss, val_loss, val_acc, best_val_acc))

            batch_iter+=1

        # evaluate on all the validation sets
        # for idx,val_loader in enumerate(val_loaders):
        #     val_acc, val_loss = validate(model, val_loader,args)
        #     # writer.add_scalar("Loss/val", val_loss, batch_iter)
        #     # writer.add_scalar("Accuracy/val", val_acc, batch_iter)
        #     val_losses[idx].append(val_loss)
        #     val_accs[idx].append(val_acc)
        #     val_iters[idx].append(batch_iter)

        # Save the best validation accuracy and the corresponding model.
    #     if best_val_acc < val_acc:
    #         best_val_acc = val_acc
    #         torch.save({
    #             'epoch': e+1,
    #             'model_state_dict': model.state_dict(),
    #             'val_acc': best_val_acc,
    #             'train_loss':train_losses,
    #             'train_iter':train_iters,
    #             'val_loss': val_loss,
    #             'val_accs':val_accs,
    #             'val_iter':val_iters
    #             }, str(args.log_name)+'.pth')

    #     # print the 
    #     print('In epoch {}, train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
    #         e, loss, val_loss, val_acc, best_val_acc))
    #     # scheduler.step()
    
    # return model,train_losses,train_iters,val_losses,val_accs,val_iters


    
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        num_samples = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            sub_idxs = torch.logical_or(target==0,target==1)
            sub_target = target[sub_idxs]
            sub_data = data[sub_idxs]
            num_samples += len(sub_target)

            # Forward
            output = model(sub_data)
            pred = output.argmax(dim=1, keepdim=True)
            val_loss += F.nll_loss(output, sub_target, reduction='sum').item()  # sum up batch loss
            correct += pred.eq(sub_target.view_as(pred)).sum().item()

        # Compute loss and accuracy
        val_loss /= num_samples
        val_acc = correct / num_samples
        # print(num_samples)
        return val_acc, val_loss


# ================================ datasets =====================================

def load_dataset(args):
    if args.dataset == "mnist":
        return load_mnist(args)

# ================================ models =====================================


def load_model(model):
    if model == "digits":
        return DigitClassifier()


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument('--model', type=str, default='digits', help='model architecture: digits')
    parser.add_argument('--dataset', type=str, default = 'mnist', help='dataset name: mnist, svhn, usps')
    parser.add_argument('--log_name', type=str, default = 'default', help='checkpoint file name')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--permuted', action='store_true', default=False,
                        help='Load permuted loaders as well')


    args = parser.parse_args()
    print(args)
    return args



# ===================================== Main =====================================
if __name__ == "__main__":
    print("=================")
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    args.device = device

    train(args)