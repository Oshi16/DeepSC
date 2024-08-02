#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

# Argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
# Define command-line arguments for input/output directories and vocabulary file
parser.add_argument('--input-data-dir', default='/content/drive/MyDrive/DeepSC_data/txt/en', type=str, 
                    help='Directory where raw text files are stored')
parser.add_argument('--output-train-dir', default='/content/drive/MyDrive/DeepSC_data/train_data.pkl', type=str, 
                    help='Path to save the processed training data')
parser.add_argument('--output-test-dir', default='/content/drive/MyDrive/DeepSC_data/test_data.pkl', type=str, 
                    help='Path to save the processed test data')
parser.add_argument('--output-vocab', default='/content/drive/MyDrive/DeepSC_data/vocab.json', type=str, 
                    help='Path to save the vocabulary file')

# Special tokens with their respective indices
SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def unicode_to_ascii(s):
    """
    Convert Unicode characters to ASCII.

    Args:
        s (str): Input string.

    Returns:
        str: ASCII encoded string.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    """
    Normalize a string by removing XML tags, extra spaces, and converting to lowercase.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """
    s = unicode_to_ascii(s)  # Convert to ASCII
    s = remove_tags(s)  # Remove HTML/XML tags
    s = re.sub(r'([!.?])', r' \1', s)  # Add space before punctuation
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)  # Remove non-alphabetic characters
    s = re.sub(r'\s+', r' ', s)  # Remove extra spaces
    s = s.lower()  # Convert to lowercase
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    """
    Filter sentences based on length.

    Args:
        cleaned (list of str): List of cleaned sentences.
        MIN_LENGTH (int): Minimum sentence length.
        MAX_LENGTH (int): Maximum sentence length.

    Returns:
        list of str: Filtered sentences.
    """
    cutted_lines = []
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines

def save_clean_sentences(sentence, save_path):
    """
    Save cleaned sentences to a file using pickle.

    Args:
        sentence (list of str): List of sentences to save.
        save_path (str): Path to save the file.
    """
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)

def process(text_path):
    """
    Process a text file to clean and filter sentences.

    Args:
        text_path (str): Path to the text file.

    Returns:
        list of str: Processed sentences.
    """
    with open(text_path, 'r', encoding='utf8') as fop:
        raw_data = fop.read()
    sentences = raw_data.strip().split('\n')
    raw_data_input = [normalize_string(data) for data in sentences]
    raw_data_input = cutted_data(raw_data_input)
    
    return raw_data_input

def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence into a list of tokens.

    Args:
        s (str): Input string.
        delim (str): Delimiter to use for tokenization.
        add_start_token (bool): Whether to add a start token.
        add_end_token (bool): Whether to add an end token.
        punct_to_keep (list of str): Punctuation to keep.
        punct_to_remove (list of str): Punctuation to remove.

    Returns:
        list of str: List of tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, token_to_idx={}, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    """
    Build vocabulary from a list of sequences.

    Args:
        sequences (list of str): List of sequences (sentences).
        token_to_idx (dict): Existing token to index mapping.
        min_token_count (int): Minimum token frequency to include.
        delim (str): Delimiter used in tokenization.
        punct_to_keep (list of str): Punctuation to keep.
        punct_to_remove (list of str): Punctuation to remove.

    Returns:
        dict: Updated token to index mapping.
    """
    token_to_count = {}

    for seq in sequences:
        seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                              punct_to_remove=punct_to_remove,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    """
    Encode tokens into indices.

    Args:
        seq_tokens (list of str): List of tokens to encode.
        token_to_idx (dict): Token to index mapping.
        allow_unk (bool): Whether to allow unknown tokens.

    Returns:
        list of int: List of token indices.
    """
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    """
    Decode indices into tokens.

    Args:
        seq_idx (list of int): List of token indices.
        idx_to_token (dict): Index to token mapping.
        delim (str): Delimiter to use for joining tokens.
        stop_at_end (bool): Whether to stop at end token.

    Returns:
        str or list of str: Decoded tokens or string.
    """
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)

def main(args):
    """
    Main function to process text files, build vocabulary, and save processed data.

    Args:
        args: Command-line arguments.
    """
    print(args.input_data_dir)
    sentences = []
    print('Preprocess Raw Text')
    for fn in tqdm(os.listdir(args.input_data_dir)):
        if not fn.endswith('.txt'): continue
        process_sentences = process(os.path.join(args.input_data_dir, fn))
        sentences += process_sentences

    # Remove duplicate sentences
    unique_sentences = {}
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences[sentence] = 0
        unique_sentences[sentence] += 1
    sentences = list(unique_sentences.keys())
    print('Number of sentences: {}'.format(len(sentences)))
    
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )

    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))

    # Save the vocabulary
    if args.output_vocab != '':
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)

    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    print('Writing Data')
    # Split data into training and testing sets
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]

    # Save the processed data
    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
