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
from models.transceiver_sparse import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/content/drive/MyDrive/DeepSC_data', type=str)
parser.add_argument('--vocab-file', default='/content/drive/MyDrive/DeepSC_data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='/content/drive/MyDrive/DeepSC_checkpoints', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int, help='Dimension of the feed-forward layer')
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, criterion, snr=None):
    """
    Validate the model on the test dataset.
    """
    test_eur = EurDataset('test', args.data_dir)
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            noise_std = SNR_to_noise(snr) if snr is not None else 0.1
            loss = val_step(net, sents, sents, noise_std, pad_idx, criterion, args.channel)
            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
    return total / len(test_iterator)

def train(epoch, args, net, optimizer, criterion, mi_net=None):
    """
    Train the model on the training dataset.
    """
    train_eur = EurDataset('train', args.data_dir)
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    total_loss = 0

    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx, optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {}; Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel)
            total_loss += loss
            pbar.set_description(
                'Epoch: {}; Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
    return total_loss / len(train_iterator)

def run_experiment(channel, args):
    """
    Run the experiment for a specific channel type (AWGN, Rayleigh, or Rician).
    """
    args.channel = channel
    channel_checkpoint_path = os.path.join(args.checkpoint_path, channel)
    
    # Ensure the checkpoint directory exists before starting
    if not os.path.exists(channel_checkpoint_path):
        os.makedirs(channel_checkpoint_path)
        print(f"Directory created: {channel_checkpoint_path}")
    
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    global pad_idx
    pad_idx = token_to_idx["<PAD>"]

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab, args.d_model, args.num_heads, args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    
    initNetParams(deepsc)

    # Check for existing checkpoint
    start_epoch = 20
    checkpoint_files = sorted([f for f in os.listdir(channel_checkpoint_path) if f.endswith('.pth')])
    if checkpoint_files:
        latest_checkpoint = os.path.join(channel_checkpoint_path, checkpoint_files[-1])
        print(f"Loading checkpoint {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        deepsc.load_state_dict(checkpoint)
        # If you saved optimizer and epoch in the checkpoint, load them as well:
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])

    training_losses = []
    validation_losses = []

    for epoch in range(args.epochs):
        avg_train_loss = train(epoch, args, deepsc, optimizer, criterion)
        training_losses.append(avg_train_loss)
        avg_val_loss = validate(epoch, args, deepsc, criterion)
        validation_losses.append(avg_val_loss)

        # Save checkpoint at each epoch
        checkpoint_filename = os.path.join(channel_checkpoint_path, f'checkpoint_{str(epoch + 1).zfill(2)}.pth')
        with open(checkpoint_filename, 'wb') as f:
            torch.save({
                'epoch': epoch,
                'model_state_dict': deepsc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Include any other states you want to save
            }, f)
        print(f"Checkpoint saved: {checkpoint_filename}")
    
    return training_losses, validation_losses

'''
        if avg_val_loss < min(validation_losses):
            # Save checkpoint
            checkpoint_filename = os.path.join(channel_checkpoint_path, f'checkpoint_{str(epoch + 1).zfill(2)}.pth')
            with open(checkpoint_filename, 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            print(f"Checkpoint saved: {checkpoint_filename}")
            '''

def plot_losses(channel, training_losses, validation_losses):
    """
    Plot the training and validation losses.
    """
    plt.figure(figsize=(10, 5))
    epochs_range = list(range(1, len(training_losses) + 1))
    plt.plot(epochs_range, training_losses, label='Training Loss')
    plt.plot(epochs_range, validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs Validation Loss ({channel})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    
    setup_seed(42)
    
    channels = ['AWGN', 'Rayleigh', 'Rician']
    for channel in channels:
        training_losses, validation_losses = run_experiment(channel, args)
        plot_losses(channel, training_losses, validation_losses)
