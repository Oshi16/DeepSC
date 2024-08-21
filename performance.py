# !usr/bin/env python
# -*- coding:utf-8 _*

import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText, beam_search_decode
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/content/drive/MyDrive/DeepSC_data', type=str)
parser.add_argument('--vocab-file', default='/content/drive/MyDrive/DeepSC_data/vocab.json', type=str)
parser.add_argument('--checkpoint-dir', default='/content/drive/MyDrive/DeepSC_checkpoints', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--channel', type=str, choices=['AWGN', 'Rayleigh', 'Rician'], help='Specify which channel to run', required=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_bleu_scores(channel, bleu_scores):
    output_path = f'bleu_scores_{channel}.json'
    with open(output_path, 'w') as f:
        json.dump(bleu_scores, f, indent=4)
    print(f"BLEU scores saved to {output_path}")

def performance(args, SNR, net, channel_type):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    bleu_score_2gram = BleuScore(0.5, 0.5, 0, 0)
    bleu_score_3gram = BleuScore(0.33, 0.33, 0.33, 0)
    bleu_score_4gram = BleuScore(0.25, 0.25, 0.25, 0.25)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score_1gram = []
    score_2gram = []
    score_3gram = []
    score_4gram = []

    bleu_scores = {'1gram': [], '2gram': [], '3gram': [], '4gram': []}

    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for i, sents in enumerate(test_iterator):
                    sents = sents.to(device)
                    target = sents

                    # Store original sentences (before noise is added)
                    original_sentences = target.cpu().numpy().tolist()
                    original_string = list(map(StoT.sequence_to_text, original_sentences))

                    # Embedding: Convert sentence indices to embeddings
                    embedded_input = net.embedding(sents)

                    # Add noise to the embedded inputs
                    noisy_input = embedded_input + torch.randn_like(embedded_input) * noise_std

                    # Now perform decoding using the noisy embeddings
                    out = beam_search_decode(net, noisy_input, noise_std, args.MAX_LENGTH, pad_idx,
                                             start_idx, channel_type, beam_width=5)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

                # Print 1 example for this SNR level
                print(f"SNR: {snr} dB, Channel: {channel_type}")
                for orig, tx, rx in zip(original_string[:1], word[:1], target_word[:1]):
                    print(f"Original:    {orig}")
                    print(f"Transmitted: {tx}")
                    print(f"Received:    {rx}")
                    print("")

            bleu_score_1gram_list = []
            bleu_score_2gram_list = []
            bleu_score_3gram_list = []
            bleu_score_4gram_list = []

            for sent1, sent2 in zip(Tx_word, Rx_word):
                bleu_score_1gram_list.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
                bleu_score_2gram_list.append(bleu_score_2gram.compute_blue_score(sent1, sent2))
                bleu_score_3gram_list.append(bleu_score_3gram.compute_blue_score(sent1, sent2))
                bleu_score_4gram_list.append(bleu_score_4gram.compute_blue_score(sent1, sent2))

            score_1gram.append(np.mean(bleu_score_1gram_list, axis=1))
            score_2gram.append(np.mean(bleu_score_2gram_list, axis=1))
            score_3gram.append(np.mean(bleu_score_3gram_list, axis=1))
            score_4gram.append(np.mean(bleu_score_4gram_list, axis=1))

            bleu_scores['1gram'].append(np.mean(bleu_score_1gram_list))
            bleu_scores['2gram'].append(np.mean(bleu_score_2gram_list))
            bleu_scores['3gram'].append(np.mean(bleu_score_3gram_list))
            bleu_scores['4gram'].append(np.mean(bleu_score_4gram_list))

    # Save BLEU scores for this channel
    save_bleu_scores(channel_type, bleu_scores)

    # Plot BLEU score vs SNR for 1, 2, 3, 4-grams
    plt.figure(figsize=(10, 6))
    plt.plot(SNR, bleu_scores['1gram'], label="1-gram BLEU")
    plt.plot(SNR, bleu_scores['2gram'], label="2-gram BLEU")
    plt.plot(SNR, bleu_scores['3gram'], label="3-gram BLEU")
    plt.plot(SNR, bleu_scores['4gram'], label="4-gram BLEU")

    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU Score')
    plt.title(f'BLEU Score vs SNR ({channel_type} Channel)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return bleu_scores

'''def performance(args, SNR, net, channel_type):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    bleu_score_2gram = BleuScore(0.5, 0.5, 0, 0)
    bleu_score_3gram = BleuScore(0.33, 0.33, 0.33, 0)
    bleu_score_4gram = BleuScore(0.25, 0.25, 0.25, 0.25)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score_1gram = []
    score_2gram = []
    score_3gram = []
    score_4gram = []

    bleu_scores = {'1gram': [], '2gram': [], '3gram': [], '4gram': []}

    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for i, sents in enumerate(test_iterator):
                    sents = sents.to(device)
                    target = sents

                    # Store original sentences (before noise is added)
                    original_sentences = target.cpu().numpy().tolist()
                    original_string = list(map(StoT.sequence_to_text, original_sentences))

                    # Generate noisy transmitted signal
                    noisy_input = sents.float() + torch.randn_like(sents.float()) * noise_std


                    out = beam_search_decode(net, noisy_input, noise_std, args.MAX_LENGTH, pad_idx,
                         start_idx, channel_type, beam_width=5)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

                # Print 1 examples for this SNR level
                print(f"SNR: {snr} dB, Channel: {channel_type}")
                for orig, tx, rx in zip(original_string[:1], word[:1], target_word[:1]):
                    print(f"Original:    {orig}")
                    print(f"Transmitted: {tx}")
                    print(f"Received:    {rx}")
                    print("")
                    
            bleu_score_1gram_list = []
            bleu_score_2gram_list = []
            bleu_score_3gram_list = []
            bleu_score_4gram_list = []

            for sent1, sent2 in zip(Tx_word, Rx_word):
                bleu_score_1gram_list.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
                bleu_score_2gram_list.append(bleu_score_2gram.compute_blue_score(sent1, sent2))
                bleu_score_3gram_list.append(bleu_score_3gram.compute_blue_score(sent1, sent2))
                bleu_score_4gram_list.append(bleu_score_4gram.compute_blue_score(sent1, sent2))

            score_1gram.append(np.mean(bleu_score_1gram_list, axis=1))
            score_2gram.append(np.mean(bleu_score_2gram_list, axis=1))
            score_3gram.append(np.mean(bleu_score_3gram_list, axis=1))
            score_4gram.append(np.mean(bleu_score_4gram_list, axis=1))

            bleu_scores['1gram'].append(np.mean(bleu_score_1gram_list))
            bleu_scores['2gram'].append(np.mean(bleu_score_2gram_list))
            bleu_scores['3gram'].append(np.mean(bleu_score_3gram_list))
            bleu_scores['4gram'].append(np.mean(bleu_score_4gram_list))

    # Save BLEU scores for this channel
    save_bleu_scores(channel_type, bleu_scores)

    # Plot BLEU score vs SNR for 1, 2, 3, 4-grams
    plt.figure(figsize=(10, 6))
    plt.plot(SNR, bleu_scores['1gram'], label="1-gram BLEU")
    plt.plot(SNR, bleu_scores['2gram'], label="2-gram BLEU")
    plt.plot(SNR, bleu_scores['3gram'], label="3-gram BLEU")
    plt.plot(SNR, bleu_scores['4gram'], label="4-gram BLEU")

    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU Score')
    plt.title(f'BLEU Score vs SNR ({channel_type} Channel)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return bleu_scores

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab, args.d_model, args.num_heads, args.dff, 0.1).to(device)

    channel = args.channel
    channel_checkpoint_path = os.path.join(args.checkpoint_dir, channel)

    # Check if the directory exists
    if not os.path.exists(channel_checkpoint_path):
        print(f"Directory does not exist: {channel_checkpoint_path}")
        exit()

    model_paths = []
    for fn in os.listdir(channel_checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of the file
        model_paths.append((os.path.join(channel_checkpoint_path, fn), idx))

    if not model_paths:
        print(f"No checkpoints found for {channel} channel. Exiting...")
        exit()

    model_paths.sort(key=lambda x: x[1])  # sort the files by the idx

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print(f'Model for {channel} channel loaded!')

    bleu_score = performance(args, SNR, deepsc, channel)
    print(f'BLEU Score for {channel} channel:', bleu_score)'''
