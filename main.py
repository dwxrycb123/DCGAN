#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('Agg') # for SSH service
from argparse import ArgumentParser
import numpy as np
import torch
from model import *
from data import *
from train import *
import os

# parser args
parser = ArgumentParser('DCGAN')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--D-epochs', type=int, default=4)
parser.add_argument('--lr', type=int, default=0.0003)
parser.add_argument('--batch-size', type=int, default=128) # 128
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--test', action='store_true')
parser.add_argument('--pretrain', type=int, default=0)

if __name__ == '__main__':
    # parse the args
    args = parser.parse_args()

    # whether to use cuda
    cuda = torch.cuda.is_available() and args.cuda

    # dataset
    train_dataset = get_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = 28 ** 2

    # cuda 
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and cuda else "cpu")
    
    G = Generator().to(device)
    D = Discriminator().to(device)

    if args.pretrain:
        G.load_state_dict(torch.load('Generator_state_dict_{}epochs.pth'.format(args.pretrain)))
        D.load_state_dict(torch.load('Discriminator_state_dict_{}epochs.pth'.format(args.pretrain)))

    if not args.test:
        train(G, D, args, train_dataloader, device)
    