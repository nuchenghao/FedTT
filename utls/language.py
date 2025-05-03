import re
import numpy as np
import torch
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)
MAX_LEN = 1000
def _one_hot(index, size):
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec
def letter_to_vec(letter):
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)
def word_to_indices(word):
    indices = []
    for c in word:
        indices.append(max(ALL_LETTERS.find(c), 0))
    return indices
def split_line(line):
    return re.findall(r"[\w']+|[.,!?;]", line)
def _word_to_index(word, indd):
    if word in indd:
        return indd[word]
    else:
        return len(indd)
def line_to_indices(line, word2id, max_words=25):
    unk_id = len(word2id)
    line_list = split_line(line)
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl
def bag_of_words(line, vocab):
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features
def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = padding_(x_batch, MAX_LEN)
    return torch.tensor(x_batch)
def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return np.array(y_batch)
