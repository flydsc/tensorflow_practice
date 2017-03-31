import numpy as np


def loaddic():
    word2id = dict()
    id2word = dict()
    with open('dict.txt', 'r') as inf:
        for idx, line in enumerate(inf.readlines()):
            temp = line.strip().split("\t")
            if "\t" not in line:
                word2id[" "] = line.strip()
            else:
                word2id[temp[1]] = temp[0]
                id2word[temp[0]] = temp[1]
    return word2id, id2word


def loadfile():
    word2id, id2word = loaddic()
    X = []
    with open('test.tsv', 'r') as infile:
        for idx, line in enumerate(infile.readlines()):
            if idx > 0:
                temp = line.strip().split('\t')
                if temp[-2] == " ":
                    X.append(temp[-1])
                else:
                    X.append(temp[-1].split(" "))

    # print len(X), len(Y)
    max_len = 0
    for i in X:
        if len(i) > max_len:
            max_len = len(i)

    X_new = []
    for x in range(len(X)):
        xx = np.zeros(max_len)
        for idx, w in enumerate(X[x]):
            # print w
            try:
                xx[idx] = word2id[w]
            except KeyError, e:
                xx[idx] = word2id["<UNK>"]
        X_new.append(xx)
    return X_new

# print loadfile()[-1][:1]
