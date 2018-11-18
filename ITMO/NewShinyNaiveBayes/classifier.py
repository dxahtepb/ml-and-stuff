# encoding: utf-8
import math
from collections import defaultdict
from typing import List

import numpy as np

from ITMO.NewShinyNaiveBayes.reader import Mail


class NaiveBayesClassifier:
    def __init__(self, lower_border=0.1, upper_border=0.9):
        self.lower_border = lower_border
        self.upper_border = upper_border
        self.prob_word_spam = defaultdict(float)
        self.prob_word_ham = defaultdict(float)
        self.spam_prob = 0
        self.ham_prob = 0
        self.word_spamicity = defaultdict(float)

    def fit(self, messages: List[Mail]):
        spam_size = sum([1 for m in messages if m.spam])
        ham_size = len(messages) - spam_size

        for message in messages:
            for word in message.all_words:
                if message.spam:
                    if word not in self.prob_word_spam:
                        self.prob_word_spam[word] = 0
                    self.prob_word_spam[word] += 1
                else:
                    if word not in self.prob_word_ham:
                        self.prob_word_ham[word] = 0
                    self.prob_word_ham[word] += 1

        for word in self.prob_word_spam:
            self.prob_word_spam[word] /= spam_size
        for word in self.prob_word_ham:
            self.prob_word_ham[word] /= ham_size

        self.spam_prob = spam_size / len(messages)
        self.ham_prob = ham_size / len(messages)

        for word in self.prob_word_spam:
            self.word_spamicity[word] = (self.prob_word_spam[word] * self.spam_prob) / \
                                        (self.prob_word_spam[word] * self.spam_prob +
                                         self.prob_word_ham[word] * self.ham_prob)

    def predict(self, messages: List[Mail], method='all', **kwargs):
        if method == 'all':
            return [self.calc_probability(message.all_words, self.word_spamicity) > kwargs['порожек']
                    for message in messages]
        elif method == 'split':
            subject_spamicity = [self.calc_probability(message.subject, self.word_spamicity)
                                 for message in messages]
            body_spamicity = [self.calc_probability(message.body, self.word_spamicity)
                              for message in messages]
            return [self.despacito2(subj_s, body_s) > kwargs['порожек']
                    for subj_s, body_s in zip(subject_spamicity, body_spamicity)]
        elif method == 'log':
            return [self.посчитай_чиселки_с_логарифмами(message.all_words) > 0
                    for message in messages]
        else:
            assert False

    def посчитай_чиселки_с_логарифмами(self, words):
        конст_с_логарифмом = np.log(self.spam_prob / self.ham_prob)

        x = sum(
            [np.log(self.prob_word_spam[word] / self.prob_word_ham[word])
             for word in words
             if self.lower_border < self.prob_word_ham[word] < self.upper_border
                and self.lower_border < self.prob_word_spam[word] < self.upper_border]
        )

        return конст_с_логарифмом + x

    def despacito2(self, subj_s, body_s):
        return (subj_s*body_s) / (subj_s*body_s + (1 - subj_s)*(1 - body_s))

    def calc_probability(self, words, source):
        prob = lambda x: 1/(1+math.exp(x))
        n = lambda x: np.log(1 - x) - np.log(x)
        word_probs = []
        for word in words:
            if word in source:
                if self.lower_border < source[word] < self.upper_border:
                    word_probs.append(n(source[word]))
        return prob(sum(word_probs))
