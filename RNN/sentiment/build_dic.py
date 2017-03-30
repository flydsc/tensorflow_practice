dic = []

with open('train.tsv', 'r') as infile:
    for line in infile.readlines():
        dic += line.strip().split("\t")[-2].split(" ")

dic = sorted(set(dic))
dic.insert(0, u'<UNK>')

with open("dict.txt", 'w') as outfile:
    for idx, ch in enumerate(dic):
        if ch is not None :
            outfile.write(str(idx) + "\t" + ch + '\n')

