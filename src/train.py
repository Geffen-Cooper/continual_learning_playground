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

def train(args):
    print("training configuration:")
    print("Model: ",args.model)
    print("Dataset:", args.dataset)


    # first load the dataset
    train_loader, val_loader, test_loader = load_dataset(args)
    # visualize_batch(train_loader)
    
    # init tensorboard
    writer = SummaryWriter()

    # create the model
    model = load_model(args.model).to(args.device)


    # create the optimizer
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_acc = 0

    model.train()
    print(args.device)
    batch_iter = 0

    for e in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            writer.add_scalar("Loss/train", loss, batch_iter)
        
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                # evaluate on the validation set
                val_acc, val_loss = validate(model, val_loader)
                writer.add_scalar("Loss/val", val_loss, batch_iter)
                writer.add_scalar("Accuracy/val", val_acc, batch_iter)

                print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f})'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss, val_loss, val_acc, best_val_acc))

            batch_iter+=1

        # evaluate on the validation set
        val_acc, val_loss = validate(model, val_loader)
        writer.add_scalar("Loss/val", val_loss, batch_iter)
        writer.add_scalar("Accuracy/val", val_acc, batch_iter)

        # Save the best validation accuracy and the corresponding model.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': e+1,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                }, 'best_'+str(args.model)+'_' + str(args.dataset)+'.pth')

        # print the 
        print('In epoch {}, train loss: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
            e, loss, val_loss, val_acc, best_val_acc))
        scheduler.step()
    
def validate(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            # Forward
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        return val_acc, val_loss


# ================================ datasets =====================================

def load_dataset(args):
    if args.dataset == "mnist":
        return load_mnist(args)
    # elif dataset == "svhn":
    #     return load_svhn()
    # elif dataset == "usps":
    #     return load_usps()

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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')


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