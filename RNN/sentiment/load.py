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
    Y = []
    with open('train.tsv', 'r') as infile:
        for idx, line in enumerate(infile.readlines()):
            if idx > 0:
                temp = line.strip().split('\t')
                if temp[-2] == " ":
                    X.append(temp[-2])
                else:
                    X.append(temp[-2].split(" "))

                Y.append(temp[-1])
    # print len(X), len(Y)
    max_len = 0
    for i in X:
        if len(i) > max_len:
            max_len = len(i)

    X_new = []
    for x in range(len(X)):
        xx = np.zeros(max_len)
        for idx, w in enumerate(X[x]):
            # print X[x]
            xx[idx] = word2id[w]
        X_new.append(xx)
    Y_new = []
    # print len(set(Y))
    l_n = len(set(Y))
    for y_ in Y:
        t_y = np.zeros(l_n)
        t_y[int(y_)] = 1
        Y_new.append(t_y)
    return l_n, max_len, len(X_new), len(word2id), X_new, Y_new

print loadfile()[2]
