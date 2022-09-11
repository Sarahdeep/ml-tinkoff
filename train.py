import argparse
import os
import pickle
import re
import random
import sys


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', nargs='?', default=sys.stdin)
    parser.add_argument('--model', type=argparse.FileType('wb'), nargs='?', default='./model.pkl')
    parser.add_argument('--seq-len', type=int, default=2)
    return parser.parse_args()


def get_raw_text():
    args = parse_cmd_line_arguments()
    result = ''
    if args.input_dir == sys.stdin:
        data = args.input_dir.read()
        data = re.sub(r'[^A-Za-zА-Яа-я]', ' ', data).lower()
        result += data
    else:
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as src:
                        data = src.read()
                        data = re.sub(r'[^A-Za-zА-Яа-я]', ' ', data).lower()
                        result += data
    return result


def tokenize():
    return tuple(filter(None, re.split(' ', get_raw_text())))


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.model = {}

    def update(self):
        n = self.n
        words = tokenize()
        for i in range(len(words) - n):
            key = words[i:i + n]
            if key not in self.model.keys():
                self.model[key] = []
            self.model[key].append(words[i + n])

    def generate_text(self, length, sentence):
        n = self.n
        q = []
        result = []
        if sentence is None:
            sentence = ''
        sentence = ' '.join(sentence)
        sentence = re.sub(r'[^A-Za-zА-Яа-я]', ' ', sentence).lower()
        words = tuple(sentence.split())
        for i in range(len(words)):
            candidates = []
            for key in self.model:
                key = tuple(key)
                if words[i::] == key[len(key) + i - len(words)::]:
                    candidates.append(key)
            if len(candidates) != 0:
                r_first_words_patterns = random.choice(candidates)
                q = list(r_first_words_patterns)
                break
        result.append(' '.join(list(words)))
        if len(q) == 0:
            words = random.choice(tuple(self.model.keys()))
            q = list(words)
            result.append(' '.join(q))
        while len(result) <= length:
            try:
                next_word = random.choice(tuple(self.model[tuple(q)]))
            except KeyError:
                next_word = self.find_similar_pattern(tuple(q))

            result.append(next_word)
            q.pop(0)
            q.append(next_word)
        return ' '.join(result)

    def find_similar_pattern(self, q: tuple):
        for i in range(1, len(q)):
            candidates = []
            for key in self.model:
                key = tuple(key)
                if q[i::] == key[i::]:
                    candidates.append(key)
            if len(candidates) != 0:
                next_word = random.choice(self.model[random.choice(candidates)])
                return next_word
        return list(random.choice(self.model))


if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    m = NgramModel(args.seq_len)
    m.update()
    pickle.dump(m,  args.model)
