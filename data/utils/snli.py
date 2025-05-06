import os
from math import ceil
import pickle
import re
import numpy as np

GLOVE_NAME = "glove.840B.300d.txt"
GLOVE_DIM = 300
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"

def read_snli(data_dir, is_train):
    def extract_text(s):
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    if is_train:
        file_name = os.path.join(data_dir, 'snli_1.0_train.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]
        file_name = os.path.join(data_dir, 'snli_1.0_dev.txt')
        with open(file_name, 'r') as f:
            for row in f.readlines()[1:]:
                rows.append(row.split('\t'))
    else:
        file_name = os.path.join(data_dir, 'snli_1.0_test.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

def construct_vocab_and_word2vec(all_data_root):
    print("========= start constructing vocab ================")
    vocab = ["<s>", "</s>"]
    snli_root = os.path.join(all_data_root , "snli")
    vocab_path = os.path.join(all_data_root, "snli", VOCAB_NAME)
    for _ in range(2):
        premises, hypotheses, labels = read_snli(snli_root, _ )
        for sentence in premises:
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
        for sentence in hypotheses:
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
    with open(vocab_path, "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)
    print(f"Loaded vocab with {len(vocab)} words.")
    print("========= start mapping vocab to vector ================")
    word_vec = {}
    glove_path = os.path.join(all_data_root, "snli", GLOVE_NAME)
    wordvec_path = os.path.join(all_data_root, "snli", WORDVEC_NAME)
    with open(glove_path, "r") as glove_file:
        for line in glove_file:
            word, vec = line.split(' ', 1)
            if word in vocab:
                word_vec[word] = np.array(list(map(float, vec.split())))
    with open(wordvec_path, "wb") as wordvec_file:
        pickle.dump(word_vec, wordvec_file)
    print(f"Found {len(word_vec)}/{len(vocab)} words with glove vectors.")
construct_vocab_and_word2vec("../")
