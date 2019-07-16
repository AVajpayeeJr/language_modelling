import codecs
from collections import defaultdict
from math import log

class ARPALM:
    def __init__(self, word2idx):
        self._word2idx = word2idx
        self._idx2word = {v:k for k,v in self._word2idx.items()}

        self._base_unigrams = {}
        self._base_bigrams = {}

        self._avg_unigram_backoff = 0
        self._avg_bigram_backoff = 0

        self._unigrams = defaultdict(lambda: [0, 0])
        self._bigrams = defaultdict(lambda: [0, 0])
        self._trigrams = defaultdict(lambda: [0, 0])

        self._bigrams_list = defaultdict(lambda: set())
        self._trigrams_list = defaultdict(lambda: set())

    def read_lm(self, lm_file):
        marker = 'start'

        unigram_cnt, bigram_cnt = 0, 0

        with codecs.open(lm_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line:
                    if marker == 'start':
                        if line == '\\1-grams:':
                            marker = 'uni'
                    elif marker == 'uni':
                        if line != '\\2-grams:':
                            try:  # catch invalid backoffs (</s>, <unk>)
                                prob, unigram, backoff = line.split()
                                try:
                                    self._base_unigrams[self._word2idx[unigram]] = 10 ** float(backoff)
                                    self._avg_unigram_backoff += 10 ** float(backoff)
                                    unigram_cnt += 1
                                except KeyError:
                                    pass
                            except ValueError:
                                pass
                        else:
                            marker = 'bi'
                    elif marker == 'bi':
                        if line != '\\3-grams:':
                            try:  # more catching
                                prob, history, word, backoff = line.split()
                                try:
                                    bigram = (self._word2idx[history], self._word2idx[word])
                                    self._base_bigrams[bigram] = 10 ** float(backoff)
                                    self._avg_bigram_backoff += 10 ** float(backoff)
                                    bigram_cnt += 1
                                except KeyError:
                                    pass
                            except ValueError:
                                pass
                        else:
                            marker = "tri"
                    elif marker == "tri":
                        continue

        self._avg_unigram_backoff /= unigram_cnt
        self._avg_bigram_backoff /= bigram_cnt

    def convert_to_ngram(self, label_probabilities):
        for sent in label_probabilities:
            for i, label_prob in enumerate(sent):
                word_idx = label_prob[0]
                prob = label_prob[1]

                self._unigrams[word_idx] = [self._unigrams[word_idx][0] + prob, self._unigrams[word_idx][1] + 1]
                if i - 1 >= 0:
                    prev_word_idx = sent[i-1][0]
                    prev_word_prob = sent[i-1][1]
                    self._bigrams[(prev_word_idx, word_idx)] = [self._bigrams[(prev_word_idx, word_idx)][0] +
                                                                (prev_word_prob * prob),
                                                                self._bigrams[(prev_word_idx, word_idx)][1] + 1]
                    self._bigrams_list[prev_word_idx].add(word_idx)
                if i - 2 >= 0:
                    prev_word_idx = sent[i - 1][0]
                    prev_word_prob = sent[i - 1][1]
                    prev_prev_word_idx = sent[i - 2][0]
                    prev_prev_word_prob = sent[i - 2][1]

                    trigram = (prev_prev_word_idx, prev_word_idx, word_idx)
                    trigram_prob = prev_prev_word_prob * prev_word_prob * prob

                    self._trigrams[trigram] = [self._trigrams[trigram][0] + trigram_prob,
                                               self._trigrams[trigram][1] + 1]
                    self._trigrams_list[(prev_prev_word_idx, prev_word_idx)].add(word_idx)

        ### unigrams
        # average
        unigram_prob_sum = 0
        for unigram in self._unigrams:
            val = self._unigrams[unigram][0] / self._unigrams[unigram][1]
            self._unigrams[unigram][0] = val
            unigram_prob_sum += val
        # norm
        for unigram in self._unigrams:
            self._unigrams[unigram][0] /= unigram_prob_sum

        ### bigrams
        # average
        for bigram in self._bigrams:
            self._bigrams[bigram][0] = self._bigrams[bigram][0] / self._bigrams[bigram][1]
        # norm
        for history in self._bigrams_list:
            if len(self._bigrams_list[history]) == 1:
                try:
                    self._bigrams[(history, self._bigrams_list[history].pop())][0] = self._base_unigrams[history]
                except KeyError:
                    self._bigrams[(history, self._bigrams_list[history].pop())][0] = self._avg_unigram_backoff
            else:
                bigram_prob_sum = 0
                for trans in self._bigrams_list[history]:
                    bigram_prob_sum += self._bigrams[(history, trans)][0]
                for trans in self._bigrams_list[history]:
                    try:
                        self._bigrams[(history, trans)][0] /= (bigram_prob_sum / self._base_unigrams[history])
                    except KeyError:
                        self._bigrams[(history, trans)][0] /= (bigram_prob_sum / self._avg_unigram_backoff)

        ### trigrams
        # average
        for trigram in self._trigrams:
            self._trigrams[trigram][0] = self._trigrams[trigram][0] / self._trigrams[trigram][1]
        # norm
        for history in self._trigrams_list:
            if len(self._trigrams_list[history]) == 1:
                trigram = (history[0], history[1], self._trigrams_list[history].pop())
                try:
                    self._trigrams[trigram][0] = self._base_bigrams[(history[0], history[1])]
                except KeyError:
                    self._trigrams[trigram][0] = self._avg_bigram_backoff
            else:
                trigram_prob_sum = 0
                for trans in self._trigrams_list[history]:
                    trigram_prob_sum += self._trigrams[(history[0], history[1], trans)][0]
                for trans in self._trigrams_list[history]:
                    try:
                        self._trigrams[(history[0], history[1], trans)][0] /= (trigram_prob_sum /
                                                                               self._base_bigrams[(history[0],
                                                                                                   history[1])])
                    except KeyError:
                        self._trigrams[(history[0], history[1], trans)][0] /= (trigram_prob_sum /
                                                                               self._avg_bigram_backoff)

    def write_arpa_format(self, file_path):
        model_file = codecs.open(file_path, "w+", encoding="utf8")

        print("\\data\\", file=model_file)
        print("ngram 1={0}".format(len(self._unigrams)), file=model_file)
        print("ngram 2={0}".format(len(self._bigrams)), file=model_file)
        print("ngram 3={0}".format(len(self._trigrams)), file=model_file)

        print("\n\\1-grams:", file=model_file)
        for unigram in sorted(self._unigrams.keys()):
            if unigram == 0:
                pass
            if unigram == 1:
                print("{0} {1}".format(log(self._unigrams[unigram][0], 10),
                                       self._idx2word[unigram]),
                      file=model_file)
            else:
                try:  # try/catch for log(0)
                    print("{0} {1} {2}".format(log(self._unigrams[unigram][0], 10),
                                               self._idx2word[unigram],
                                               log(self._base_unigrams[unigram], 10)), file=model_file)
                except ValueError:
                    print("{0} {1} {2}".format(-99,
                                               self._idx2word[unigram],
                                               log(self._base_unigrams[unigram], 10)),
                          file=model_file)  # ARPA standard substitution for log(0) is -99 according to documentation
                except KeyError:  # catch for out-of-model
                    print("{0} {1} {2}".format(log(self._unigrams[unigram][0], 10),
                                               self._idx2word[unigram],
                                               log(self._avg_unigram_backoff, 10)),
                          file=model_file)

        print("\n\\2-grams:", file=model_file)
        for bigram in sorted(self._bigrams.keys()):
            try:
                print("{0} {1} {2} {3}".format(log(self._bigrams[bigram][0], 10),
                                               self._idx2word[bigram[0]],
                                               self._idx2word[bigram[1]],
                                               log(self._base_bigrams[(bigram[0], bigram[1])], 10)),
                      file=model_file)
            except ValueError:
                print("{0} {1} {2}".format(-99,
                                           self._idx2word[bigram[0]],
                                           self._idx2word[bigram[1]],
                                           log(self._base_bigrams[(bigram[0], bigram[1])], 10)),
                      file=model_file)
            except KeyError:
                print("{0} {1} {2} {3}".format(log(self._bigrams[bigram][0], 10),
                                               self._idx2word[bigram[0]],
                                               self._idx2word[bigram[1]],
                                               log(self._avg_bigram_backoff, 10)),
                      file=model_file)

        print("\n\\3-grams:", file=model_file)
        for trigram in sorted(self._trigrams.keys()):
            try:
                print("{0} {1} {2} {3}".format(log(self._trigrams[trigram][0], 10),
                                               self._idx2word[trigram[0]],
                                               self._idx2word[trigram[1]],
                                               self._idx2word[trigram[2]]), file=model_file)
            except ValueError:
                print("{0} {1} {2} {3}".format(-99,
                                               self._idx2word[trigram[0]],
                                               self._idx2word[trigram[1]],
                                               self._idx2word[trigram[2]]),
                      file=model_file)

        print("\n\\end\\", file=model_file)
        model_file.close()
