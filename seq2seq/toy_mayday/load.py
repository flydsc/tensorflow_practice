# -*- coding: utf-8 -*-
import codecs
import numpy as np

idx2ch = dict()
ch2idx = dict()


with codecs.open('dict.txt', 'r', 'utf-8') as infile:
    for line in infile.readlines():
        idx, ch = line.strip().split('\t')
        idx2ch[int(idx)] = ch
        ch2idx[ch] = int(idx)
    print max(idx2ch)

def seq():
    sequences = []
    with codecs.open('data.txt', 'r', 'utf-8') as infile:
        for line in infile.readlines():
            num_line = [ch2idx[ch] for ch in line.strip().replace(" ", "")]
            sequences.append(num_line)
    return sequences


def test():
    sequences = []
    testline = u"有谁能听见"
    sequences.append([ch2idx[ch] for ch in testline])
    return sequences


def getch(idx_seq):
    print idx_seq.tolist()
    return u"".join(idx2ch[idx] for idx in idx_seq.tolist()[0])

