import codecs

text = []
dic = []

with codecs.open('data1.txt', 'r', 'utf-8') as infile:
    for line in infile.readlines():
        text += line.replace(" ", "").replace("\n", "|")

dic = list(set(text))
dic = sorted(dic)
dic.insert(0, u'PAD')
dic.insert(1, u'<EOS>')

with codecs.open("dict.txt", 'w', 'utf-8') as outfile:
    for idx, ch in enumerate(dic):
        if ch != "|":
            outfile.write(str(idx) + "\t" + ch + '\n')

