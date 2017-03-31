fil = []
id = []
with open("test.tsv", "r") as inid:
    for idx, line in enumerate(inid.readlines()):
        if idx > 0:
            id.append(line.strip().split('\t')[0])
#
with open("result.txt", "r") as infile:
    for line in infile.readlines():
        temp = line.strip().replace("[", "").replace("]","").split(", ")
        fil.extend(temp)
# print len(fil)

with open("sub.txt", "w") as out:
    out.write("PhraseId,Sentiment\n")
    for idx, num in zip(id, fil):
        out.write(str(idx)+','+str(num)+"\n")
