import os
import pickle
import random
import re

import torch
from torch.utils.data import Dataset

MASK_WORD = '_'
ALPHABET = 'aąbcćdeęfghijklłmnńoóprsśtuwyxzźż'


class CorpusPreprocessor:
    def __init__(self):
        self.words = set()
        self.alphabet = ALPHABET

    def fit_corpus(self, texts):
        words = [word for sentence in texts for word in sentence.split(" ")]
        self.words = list(set(words))

    def transform_text(self, text):
        text = re.sub("[^ " + self.alphabet + "]", " ", text.lower())
        text = re.sub("\\s+", " ", text)
        text = re.sub("^\\s", "", text)
        text = re.sub("\\s$", "", text)
        return text

    def mask_text(self, text):
        text = text.split(" ")
        idx = random.randrange(len(text))
        orig_word = text[idx]
        text[idx] = MASK_WORD
        truthy = random.randrange(2)
        if truthy:
            return " ".join(text), orig_word, 1
        else:
            r = random.choice(self.words)
            while r == orig_word:
                r = random.choice(self.words)
            return " ".join(text), r, 0


class MaskedSentencesDataset(Dataset):
    def __init__(self, pickled_path, device, cache=True):
        _, filename = os.path.split(pickled_path)
        filename = "./data/.cache/" + filename
        if cache and os.path.isfile(filename):
            print("Found cached data.")
            with open(filename, "rb") as file:
                data = pickle.load(file)
            self.sentences = data['sentences'].long().to(device)
            self.flags = data['flags'].long().to(device)
            self.words = data['words'].long().to(device)
        else:
            print("Not using cached data or cached data not found.")
            with open(pickled_path, "rb") as file:
                data = pickle.load(file)
            n = Numerizer()
            data = n.numerize_texts(data)

            max_word_len = max([max(len(word), len(masked)) for sentence, masked, _ in data for word in sentence])

            max_sentence_len = max([len(sentence) for sentence, _, _ in data])
            padded = torch.zeros((len(data), max_sentence_len, max_word_len)).int().to(device)

            for s_idx, (sentence, _, _) in enumerate(data):
                for w_idx, word in enumerate(sentence):
                    padded[s_idx, w_idx, :len(word)] = torch.IntTensor(word).to(device)
            self.sentences = padded

            padded = torch.zeros((len(data), max_word_len)).int().to(device)
            for idx, (_, word, _) in enumerate(data):
                padded[idx, :len(word)] = torch.IntTensor(word).to(device)
            self.words = padded
            self.flags = torch.LongTensor([f for _, _, f in data])
            with open(filename, "wb+") as file:
                pickle.dump({'sentences': self.sentences, 'words': self.words, 'flags': self.flags}, file, protocol=4)
            self.sentences = self.sentences.long().to(device)
            self.words = self.words.long().to(device)
            self.flags = self.flags.to(device)

    def __len__(self):
        return self.flags.shape[0]

    def __getitem__(self, item):
        # returns (masked_sentence, removed_or_random_word, positive_negative)
        return self.sentences[item], self.words[item], self.flags[item]


class Numerizer():
    def __init__(self):
        self.alphabet = ALPHABET
        self.char2num = {c: i+1 for i, c in enumerate(self.alphabet)}
        self.char2num['_'] = len(self.alphabet) + 1
        self.num2char = {i: c for c, i in self.char2num.items()}

    def numerize_texts(self, texts):
        return [(self.numerize_sentence(s), self.numerize_word(w), x) for (s, w, x) in texts]

    def alphabet_size(self):
        return len(self.alphabet)

    def numerize_word(self, word):
        return [self.char2num[c] for c in word]

    def numerize_sentence(self, sentence):
        return [self.numerize_word(word) for word in sentence.split(" ")]

    def denumerize_word(self, word):
        return [self.num2char[i] for i in word]

    def denumerize_sentence(self, sentence):
        return [self.denumerize_word(word) for word in sentence]

    def denumerize_sentences(self, sentences):
        return [self.denumerize_sentence(s) for s in sentences]

    def denumerize_texts(self, texts):
        return [(self.denumerize_sentence(s), self.denumerize_word(w), x) for (s, w, x) in texts]
