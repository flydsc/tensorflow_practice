import helpers
import codecs

sequences = []
idx2ch = dict()
ch2idx = dict()


with codecs.open('dict.txt', 'r', 'utf-8') as infile:
    for line in infile.readlines():
        idx, ch = line.strip().split('\t')
        idx2ch[idx] = ch
        ch2idx[ch] = idx




