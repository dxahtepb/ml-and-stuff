from ITMO.NaiveBayes.Dataset import *
from ITMO.NaiveBayes.Metrics import Metrics
import math
import numpy as np
import copy

class BayesClassifier:

    all_prob = {"spam":{}, "ham": {}}
    head_prob = {}
    body_prob = {}
    upper_border = 0.9
    lower_border = 0.1

    def __init__(self, data:Dataset, exclude:int):
        self.data = data
        self.exclude = exclude

    def fit(self):
        all_spam = 0
        all_ham = 0
        spam_freq = {}
        ham_freq = {}
        for batch in range(0, len(self.data.Spam)):
            if batch != self.exclude:
                all_ham += len(self.data.Ham[batch])
                all_spam += len(self.data.Spam[batch])
                for msg in self.data.Spam[batch]:
                    for word in msg.words:
                        if word in spam_freq:
                            spam_freq[word] += 1
                        else:
                            spam_freq[word] = 1
                for msg in self.data.Ham[batch]:
                    for word in msg.words:
                        if word in ham_freq:
                            ham_freq[word] += 1
                        else:
                            ham_freq[word] = 1

        for ham in ham_freq:
            ham_freq[ham] /= all_ham
        for spam in spam_freq:
            spam_freq[spam] /= all_spam
        w_spam = {}
        for w in ham_freq:
            spam_f = spam_freq[w] if w in spam_freq else 0
            w_spam[w] = spam_f / (spam_f + ham_freq[w])
        for w in spam_freq:
            ham_f = ham_freq[w] if w in ham_freq else 0
            w_spam[w] = spam_freq[w] / (ham_f + spam_freq[w])
        self.all_prob["spam"] = w_spam.copy()
        self.all_prob["ham"] = ham_freq.copy()

    def calc_probability(self, words, source):
        prob = lambda x: 1/(1+math.exp(x))
        n = lambda x: np.log(1 - x) - np.log(x)
        word_probs = []
        for word in words:
            if word in source:
                if self.lower_border < source[word] < self.upper_border:
                    word_probs.append(n(source[word]))
        return prob(sum(word_probs))

    def fit_validate(self, threshhold = 0.5, Metric = Metrics.accuracy):
        probs = {}
        max_metric = 0
        for exclude in range(0, len(self.data.Spam)):
            classes = []
            self.exclude = exclude
            self.fit()
            new_classes = self.test_batch(self.data.Spam[exclude])
            valid_classes = [1 for i in range(len(self.data.Spam[exclude]))]
            for elem in new_classes:
                if elem > threshhold:
                    classes.append(1)
                else:
                    classes.append(0)
            ham_classes = self.test_batch(self.data.Ham[exclude])
            for elem in ham_classes:
                if elem > threshhold:
                    classes.append(1)
                else:
                    classes.append(0)
            valid_classes.extend([0 for i in range(len(self.data.Ham[exclude]))])
            cur_metric = Metric(valid_classes, classes)
            if cur_metric > max_metric:
                probs = copy.deepcopy(self.all_prob)
                max_metric = cur_metric
        self.all_prob = copy.deepcopy(probs)
        return max_metric

    def test_batch(self, batch):
        classes = []
        for msg in batch:
            classes.append(self.calc_probability(msg.words, self.all_prob["spam"]))
        return classes

    def split_fit(self):
        all_spam = 0
        all_ham = 0
        spam_freq = {}
        ham_freq = {}
        for batch in range(0, len(self.data.Spam)):
            if batch != self.exclude:
                all_ham += len(self.data.Ham[batch])
                all_spam += len(self.data.Spam[batch])
                for msg in self.data.Spam[batch]:
                    for word in msg.headerWords:
                        if word in spam_freq:
                            spam_freq[word] += 1
                        else:
                            spam_freq[word] = 1
                for msg in self.data.Ham[batch]:
                    for word in msg.headerWords:
                        if word in ham_freq:
                            ham_freq[word] += 1
                        else:
                            ham_freq[word] = 1

        for ham in ham_freq:
            ham_freq[ham] /= all_ham
        for spam in spam_freq:
            spam_freq[spam] /= all_spam
        w_spam = {}
        for w in ham_freq:
            spam_f = spam_freq[w] if w in spam_freq else 0
            w_spam[w] = spam_f / (spam_f + ham_freq[w])
        for w in spam_freq:
            ham_f = ham_freq[w] if w in ham_freq else 0
            w_spam[w] = spam_freq[w] / (ham_f + spam_freq[w])
        self.head_prob = w_spam.copy()
        all_spam = 0
        all_ham = 0
        spam_freq = {}
        ham_freq = {}
        for batch in range(0, len(self.data.Spam)):
            if batch != self.exclude:
                all_ham += len(self.data.Ham[batch])
                all_spam += len(self.data.Spam[batch])
                for msg in self.data.Spam[batch]:
                    for word in msg.bodyWords:
                        if word in spam_freq:
                            spam_freq[word] += 1
                        else:
                            spam_freq[word] = 1
                for msg in self.data.Ham[batch]:
                    for word in msg.bodyWords:
                        if word in ham_freq:
                            ham_freq[word] += 1
                        else:
                            ham_freq[word] = 1

        for ham in ham_freq:
            ham_freq[ham] /= all_ham
        for spam in spam_freq:
            spam_freq[spam] /= all_spam
        w_spam = {}
        for w in ham_freq:
            spam_f = spam_freq[w] if w in spam_freq else 0
            w_spam[w] = spam_f / (spam_f + ham_freq[w])
        for w in spam_freq:
            ham_f = ham_freq[w] if w in ham_freq else 0
            w_spam[w] = spam_freq[w] / (ham_f + spam_freq[w])
        self.body_prob = w_spam.copy()

    def calc_mass_probability(self, probs):
        prob = lambda x: 1 / (1 + math.exp(x))
        n = lambda x: np.log(1 - x) - np.log(x)
        word_probs = []
        for probability in probs:
            if self.lower_border < probability < self.upper_border:
                word_probs.append(n(probability))
        return prob(sum(word_probs)) if len(word_probs) > 1 else prob(word_probs[0])

    def split_test_batch(self, batch):
        classes = []
        for msg in batch:
            classes.append(self.calc_mass_probability([self.calc_probability(msg.headerWords, self.head_prob),
                                                 self.calc_probability(msg.bodyWords, self.body_prob)]))
        return classes

    def split_fit_validate(self, threshhold = 0.5, Metric = Metrics.accuracy):
        head_probs = {}
        body_probs = {}
        max_metric = 0
        for exclude in range(0, len(self.data.Spam)):
            classes = []
            self.exclude = exclude
            self.split_fit()
            new_classes = self.split_test_batch(self.data.Spam[exclude])
            valid_classes = [1 for i in range(len(self.data.Spam[exclude]))]
            for elem in new_classes:
                if elem > threshhold:
                    classes.append(1)
                else:
                    classes.append(0)
            ham_classes = self.split_test_batch(self.data.Ham[exclude])
            for elem in ham_classes:
                if elem > threshhold:
                    classes.append(1)
                else:
                    classes.append(0)
            valid_classes.extend([0 for i in range(len(self.data.Ham[exclude]))])
            cur_metric = Metric(valid_classes, classes)
            if cur_metric > max_metric:
                head_probs = copy.deepcopy(self.head_prob)
                body_probs = copy.deepcopy(self.body_prob)
                max_metric = cur_metric
        self.head_prob = copy.deepcopy(head_probs)
        self.body_prob = copy.deepcopy(body_probs)
        return max_metric