# -*- coding: utf-8 -*-

import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

# Argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/content/drive/MyDrive/DeepSC_data', type=str, 
                    help='Directory where the dataset is stored')
parser.add_argument('--vocab-file', default='/content/drive/MyDrive/DeepSC_data/vocab.json', type=str, 
                    help='Path to the vocabulary file')
parser.add_argument('--checkpoint-path', default='/content/drive/MyDrive/DeepSC_checkpoints/deepsc-Rayleigh-SNR0-18-lr5e-5', type=str, 
                    help='Directory to save model checkpoints')
parser.add_argument('--channel', default='Rayleigh', type=str, 
                    help='Communication channel type (AWGN, Rayleigh, or Rician)')
parser.add_argument('--MAX-LENGTH', default=30, type=int, 
                    help='Maximum length of sentences')
parser.add_argument('--MIN-LENGTH', default=4, type=int, 
                    help='Minimum length of sentences')
parser.add_argument('--d-model', default=128, type=int, 
                    help='Dimension of the model')
parser.add_argument('--dff', default=512, type=int, 
                    help='Dimension of the feed-forward layer')
parser.add_argument('--num-layers', default=4, type=int, 
                    help='Number of layers in the model')
parser.add_argument('--num-heads', default=8, type=int, 
                    help='Number of attention heads')
parser.add_argument('--batch-size', default=128, type=int, 
                    help='Batch size for training')
parser.add_argument('--epochs', default=80, type=int, 
                    help='Number of training epochs')

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    """
    Validate the model on the test dataset.

    Args:
        epoch (int): The current epoch number.
        args: Command-line arguments.
        net: The model to validate.

    Returns:
        float: Average validation loss.
    """
    # Load the test dataset
    test_eur = EurDataset('test', args.data_dir)
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    
    # Set model to evaluation mode
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            # Compute validation loss
            loss = val_step(net, sents, sents, 0.1, pad_idx, criterion, args.channel)
            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    # Return average loss
    return total / len(test_iterator)

def train(epoch, args, net, mi_net=None):
    """
    Train the model on the training dataset.

    Args:
        epoch (int): The current epoch number.
        args: Command-line arguments.
        net: The model to train.
        mi_net: Mutual information network (optional).
    """
    # Load the training dataset
    train_eur = EurDataset('train', args.data_dir)
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    # Generate random noise standard deviation for channel
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))

    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # Compute mutual information and training loss
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx, optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {}; Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            # Compute training loss
            loss = train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {}; Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

if __name__ == '__main__':
    args = parser.parse_args()
    
    """ Preparing the dataset """
    # Load the vocabulary file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ Define optimizer and loss function """
    # Initialize the DeepSC model
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    
    # Initialize the mutual information network (if applicable)
    mi_net = Mine().to(device)
    
    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=5e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    
    # Initialize network parameters
    initNetParams(deepsc)

    # Training loop
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10
        
        # Train the model
        train(epoch, args, deepsc)
        
        # Validate the model
        avg_acc = validate(epoch, args, deepsc)

        # Save the model checkpoint if it improves
        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []
