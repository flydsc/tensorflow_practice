import config
import codecs


configs = config.para_config()


class Data():
    def __init__(self):
        self.seq_length = configs.seq_length
        self.batch_size = configs.batch_size
        self.data = []
        with codecs.open('sample.txt', 'r', 'utf-8') as filein:
            for line in filein.readlines():
                if len(line.strip()) > 2:
                    self.data.append(line.strip())
        self.total_len = len("".join(self.data))  # total data length
        self.words = list(set("".join(self.data)))
        self.words.sort()
        self.vocab_size = len(self.words)  # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}

        # pointer position to generate current batch
        self._pointer = 0

    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, id):
        return self.id2char_dict[id]

    def save_metadata(self, file):
        with open(file, 'w') as f:
            f.write('id\tchar\n')
            for i in range(self.vocab_size):
                c = self.id2char(i)
                f.write('{}\t{}\n'.format(i, c))

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len:
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer + 1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length  # update pointer position

            bx = [self.char2id(c) for c in bx]
            by = [self.char2id(c) for c in by]

            x_batches.append(bx)
            y_batches.append(by)
        return x_batches, y_batches
