import codecs

text = []
dic = []

with codecs.open('data1.txt', 'r', 'utf-8') as infile:
    for line in infile.readlines():
        text += line.replace(" ", "").replace("\n", "|")

dic = list(set(text))

with codecs.open("dict.txt", 'w', 'utf-8') as outfile:
    for idx, ch in enumerate(dic):
        if ch != "|":
            outfile.write(str(idx + 2) + "\t" + ch + '\n')

