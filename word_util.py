import numpy as np
import os
import jieba
import pandas as pd


class WordUtil():
    UNK_TAG = '<unk>'
    START_TAG = '<s>'
    END_TAG = '</s>'
    UNK = 0
    START = 1
    END = 2
    vocab_file_path = os.path.join('data', 'generated', 'vocab.txt')
    csv_file_path = os.path.join('data', 'generated', 'data.csv')

    def __init__(self, *word_files):
        self.word_files = word_files
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.START_TAG: self.START,
            self.END_TAG: self.END
        }
        if os.path.exists(self.vocab_file_path):
            self.load_dict()
            # jieba.load_userdict(self.vocab_file_path)
        else:
            self.build_dict()

    def load_dict(self):
        with open(self.vocab_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                if line == self.UNK_TAG \
                        or line == self.START_TAG \
                        or line == self.END_TAG:
                    continue
                self.dict[line] = len(self.dict)

    def build_dict(self):
        ''' Build words dictionary from files '''
        count = {}
        for word_file in self.word_files:
            with open(word_file, encoding='utf-8') as f:
                content = f.read()
                word_list = jieba.cut(content)
                #word_list = list(content)
                for w in word_list:
                    if w == '\n':
                        continue
                    if w not in count:
                        count[w] = 0
                    count[w] += 1

        words = sorted(count.items(), key=lambda x: x[1], reverse=True)
        for k, _ in words:
            self.dict[k] = len(self.dict)

    def word_to_index(self, word):
        if word in self.dict:
            return self.dict[word]
        return self.UNK

    def index_to_word(self, index):
        for k, v in self.dict.items():
            if index == v:
                return k
        return self.UNK_TAG

    def transform(self, sentence):
        index = []
        for w in jieba.lcut(sentence):
            if w == '\n':
                continue
            index.append(self.word_to_index(w))
        return index

    def reverse_transform(self, index):
        sentence = ''
        for i in index:
            if i == self.UNK or i == self.START or i == self.END:
                continue
            sentence += self.index_to_word(i)
        return sentence

    def get_index_array(self):
        src_tar_index = []
        for word_file in self.word_files:
            with open(word_file, encoding='utf-8') as f:
                index_list = []
                for line in f.readlines():
                    line_index = self.transform(line)
                    index_list.append(line_index)
                src_tar_index.append(index_list)
        return src_tar_index

    def generate_vocab(self):
        assert self.dict is not None
        with open(self.vocab_file_path, 'w', encoding='utf-8') as f:
            for w in self.dict.keys():
                f.write(w + '\n')

    def generate_csv(self):
        if os.path.exists(self.csv_file_path):
            return

        result = self.get_index_array()

        question = [self._list_to_str(l) for l in result[0]]
        answer = [self._list_to_str(l) for l in result[1]]

        dataframe = pd.DataFrame({'question': question, 'answer': answer})
        dataframe.to_csv(self.csv_file_path, index=False, sep=',')

    def _list_to_str(self, li):
        _str = ''
        for i in li:
            _str += str(i) + ' '
        return _str.strip()


if __name__ == '__main__':
    a = WordUtil('data/origin/question.txt', 'data/origin/answer.txt')
    print(a.get_index_array()[0][0])