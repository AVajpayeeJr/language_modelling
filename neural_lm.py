import argparse
import codecs
from collections import defaultdict
import logging
from models.char import RNNLM
from models.word_class import WordClassRNNLM
from models.arpa import ARPALM
import numpy as np
import os
import yaml

__author__ = 'avijitv@uw.edu'


def read_sentence_data(file_path):
    """ Returns sentences, size of biggest word (number of chars), size of biggest sentence (number of words)

    Returns:
        sentences (:obj:'list' of :obj: 'str')
        max_word_length (int)
        max_sentence_length (int)
    """
    sentences = []
    max_sentence_length = 0
    max_word_length = 0
    with codecs.open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line:
                tokens = line.split()
                max_sentence_length = max(max_sentence_length, len(tokens))
                max_word_length = max(max_word_length, max(len(t) for t in tokens))
                sentences.append(tokens)
    return sentences, max_word_length, max_sentence_length


def read_classes_data(file_path):
    """ Returns a mapping of words to their Brown Cluster indices and number of clusters.

    Returns:
        word2class_idx (:obj:dict of str to int)
        num_classes (int)
    """
    class_path2class_idx = {}
    word2class_idx = {}
    class_idx2word = defaultdict(list)
    class_idx_cnt = 1
    with codecs.open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line:
                class_path = line.split()[0]
                word = line.split()[1]

                if class_path not in class_path2class_idx:
                    class_path2class_idx[class_path] = class_idx_cnt
                    class_idx_cnt += 1
                class_idx = class_path2class_idx[class_path]
                word2class_idx[word] = class_idx
                class_idx2word[class_idx].append(word)
    logging.debug('Entries in class_path2class_idx: {}'.format(len(class_path2class_idx)))
    logging.debug('Entries in word2class_idx: {}'.format(len(word2class_idx)))
    logging.debug('Entries in class_idx2word: {}'.format(len(class_idx2word)))
    logging.debug('class_idx_cnt: {}'.format(class_idx_cnt))

    class_idx2word[0] = []
    num_classes = len(class_idx2word)

    return word2class_idx, class_idx2word, num_classes


def build_vocab(sentences, level='word'):
    vocab = {'<pad>': 0, '<unk>': 1}
    idx_cnt = 2
    for s in sentences:
        for word in s:
            if level == 'word':
                if word == '<unk>':
                    pass
                elif word not in vocab:
                    vocab[word] = idx_cnt
                    idx_cnt += 1
            elif level == 'char':
                for c in word:
                    if c not in vocab:
                        vocab[c] = idx_cnt
                        idx_cnt += 1
    return vocab


def build_word_model_dataset(sentences, word_vocab, max_sent_len):
    x_word = []
    y_word = []
    for s in sentences:
        sentence_word_seq = []
        for w in s:
            try:
                sentence_word_seq.append(word_vocab[w])
            except KeyError:
                sentence_word_seq.append(word_vocab['<unk>'])

        # Padding word sequence to max_sent_len
        sentence_word_seq = sentence_word_seq + [0] * (max_sent_len - len(sentence_word_seq))

        x_word.append(sentence_word_seq)
        y_word.append(sentence_word_seq[1:] + [0])
    x_word = np.array(x_word, dtype=np.int32)
    y_word = np.array(y_word, dtype=np.int32)
    y_word = np.expand_dims(y_word, -1)
    return x_word, y_word


def build_char_model_dataset(sentences, char_vocab, max_sent_len, max_word_len):
    x_char = []

    for s in sentences:
        sentence_char_seq = []

        for w in s:
            word_char_seq = []
            for c in w:
                try:
                    word_char_seq.append(char_vocab[c])
                except KeyError:
                    word_char_seq.append(char_vocab['<unk>'])

            # Padding char sequence of word to max_word_len
            word_char_seq = word_char_seq + [0] * (max_word_len - len(word_char_seq))
            sentence_char_seq.append(word_char_seq)

        # Padding sentence char sequence to max_sent_len
        sentence_char_seq = sentence_char_seq + [[0]*max_word_len for _ in
                                                 range(max_sent_len - len(sentence_char_seq))]

        x_char.append(sentence_char_seq)
    x_char = np.array(x_char, dtype=np.int32)
    return x_char


def build_class_model_dataset(sentences, word2class_idx, class_idx2word, word_vocab, max_sent_len):
    x_class = []

    for s in sentences:
        sentence_class_seq = []

        for w in s:
            class_idx = word2class_idx[w]
            sentence_class_seq.append(class_idx)

        # Padding class sequence to max_sent_len
        sentence_class_seq = sentence_class_seq + [0] * (max_sent_len - len(sentence_class_seq))
        x_class.append(sentence_class_seq)

    x_class = np.array(x_class, dtype=np.int32)

    class_one_hot_matrix = np.zeros((len(class_idx2word), len(class_idx2word)), dtype=np.int32)
    class_membership_matrix = np.zeros((len(class_idx2word), len(word_vocab)), dtype=np.int32)
    for class_idx in class_idx2word:
        class_one_hot_matrix[class_idx][class_idx] = 1
        for word in class_idx2word[class_idx]:
            word_idx = word_vocab[word]
            class_membership_matrix[class_idx][word_idx] = 1

    return x_class, class_one_hot_matrix, class_membership_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', help='Language Code (Directory for train/test/val data)')
    parser.add_argument('--config', help='Config File')
    parser.add_argument('--type', help='word/char/word_char/word_class')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes when using word_class model')
    parser.add_argument('--output_base_dir', default='saved_models', help='Base Directory for storing models')
    parser.add_argument('--input_ngram_lm', default=None,
                        help='Input N-Gram LM to use as base for approximating')
    parser.add_argument('--output_ngram_lm', default=None,
                        help='Output file to write Approximated NGram LM in ARPA Format')
    parser.add_argument('--debug', action='store_true', default=True, help='Run with DEBUG logging level')
    args = parser.parse_args()

    print(os.environ['CUDA_AVAILABLE_DEVICES'])
    if args.debug:
        logging.basicConfig(format='%(levelname)s:%(funcName)s:%(lineno)s:\t%(message)s', level=logging.DEBUG)

    with open(args.config, 'r') as infile:
        config = yaml.load(infile)

    train_file = args.language + '/' + 'train.txt'
    val_file = args.language + '/' + 'valid.txt'
    test_file = args.language + '/' + 'test.txt'

    ### Read input train / test / val files
    train_sentences, train_max_word_len, train_max_sent_len = read_sentence_data(file_path=train_file)
    val_sentences, val_max_word_len, val_max_sent_len = read_sentence_data(file_path=val_file)
    test_sentences, test_max_word_len, test_max_sent_len = read_sentence_data(file_path=test_file)

    max_word_len = max(train_max_word_len, val_max_word_len) + 5
    max_sent_len = max(train_max_sent_len, val_max_sent_len) + 5

    ### Build Word Model Datasets
    word_vocab = build_vocab(train_sentences, level='word')
    logging.debug('Size Word Vocab: {0}'.format(len(word_vocab)))

    # Train
    word_x_train, y_train = build_word_model_dataset(sentences=train_sentences,
                                                     word_vocab=word_vocab,
                                                     max_sent_len=max_sent_len)
    logging.debug('Train Word X: {0}'.format(word_x_train.shape))
    logging.debug('Train Y: {0}'.format(y_train.shape))

    # Val
    word_x_val, y_val = build_word_model_dataset(sentences=val_sentences,
                                                 word_vocab=word_vocab,
                                                 max_sent_len=max_sent_len)
    logging.debug('Val Word X: {0}'.format(word_x_val.shape))
    logging.debug('Val Y: {0}'.format(y_val.shape))

    # Test
    word_x_test, y_test = build_word_model_dataset(sentences=test_sentences,
                                                   word_vocab=word_vocab,
                                                   max_sent_len=max_sent_len)
    logging.debug('Test Word X: {0}'.format(word_x_test.shape))
    logging.debug('Test Y: {0}'.format(y_test.shape))

    ### Build Char Model Datasets
    char_vocab = build_vocab(train_sentences, level='char')
    logging.debug('Size Char Vocab: {0}'.format(len(char_vocab)))

    # Train
    char_x_train = build_char_model_dataset(sentences=train_sentences,
                                            char_vocab=char_vocab,
                                            max_sent_len=max_sent_len,
                                            max_word_len=max_word_len)
    logging.debug('Train Char X: {0}'.format(char_x_train.shape))

    # Val
    char_x_val = build_char_model_dataset(sentences=val_sentences,
                                          char_vocab=char_vocab,
                                          max_sent_len=max_sent_len,
                                          max_word_len=max_word_len)
    logging.debug('Val Char X: {0}'.format(char_x_val.shape))

    # Test
    char_x_test = build_char_model_dataset(sentences=test_sentences,
                                           char_vocab=char_vocab,
                                           max_sent_len=max_sent_len,
                                           max_word_len=max_word_len)
    logging.debug('Test Char X: {0}'.format(char_x_test.shape))

    save_dir = args.output_base_dir + '/' + args.language + '/neural/' + args.type
    if args.type == 'word_class':
        ### Build Class Model Datasets
        class_file = args.language + '/' + str(args.num_classes) + '_clusters.txt'
        word2class_idx, class_idx2word, num_classes = read_classes_data(class_file)
        logging.debug('Number of classes found: {}'.format(num_classes))

        # Train
        class_x_train, class_one_hot_matrix, class_membership_matrix = build_class_model_dataset(sentences=train_sentences,
                                                                                                 word2class_idx=word2class_idx,
                                                                                                 class_idx2word=class_idx2word,
                                                                                                 word_vocab=word_vocab,
                                                                                                 max_sent_len=max_sent_len)
        logging.debug('Train Class One-Hot Matrix Shape: {}'.format(class_one_hot_matrix.shape))
        logging.debug('Train Class Membership Matrix Shape: {}'.format(class_membership_matrix.shape))
        logging.debug('Train Class X: {0}'.format(class_x_train.shape))

        # Val
        class_x_val, _, _ = build_class_model_dataset(sentences=val_sentences,
                                                      word2class_idx=word2class_idx,
                                                      class_idx2word=class_idx2word,
                                                      word_vocab=word_vocab,
                                                      max_sent_len=max_sent_len)
        logging.debug('Val Class X: {0}'.format(class_x_val.shape))

        # Test
        class_x_test, _, _ = build_class_model_dataset(sentences=test_sentences,
                                                       word2class_idx=word2class_idx,
                                                       class_idx2word=class_idx2word,
                                                       word_vocab=word_vocab,
                                                       max_sent_len=max_sent_len)
        logging.debug('Test Class X: {0}'.format(class_x_test.shape))

        word_class_rnn_lm = WordClassRNNLM(max_seq_len=max_sent_len,
                                           word_vocab_size=len(word_vocab),
                                           class_vocab_size=num_classes,
                                           class_membership_weights=class_membership_matrix,
                                           class_one_hot_weights=class_one_hot_matrix,
                                           save_dir=save_dir,
                                           config=config)
        print('Training Word + Class Level Model')
        print('----------')
        train_class_y = np.expand_dims(class_x_train, -1)
        logging.debug('Train Class Y: {}'.format(train_class_y.shape))
        val_class_y = np.expand_dims(class_x_val, -1)
        logging.debug('Val Class Y: {}'.format(val_class_y.shape))
        word_class_rnn_lm.train(train_x=[class_x_train, word_x_train],
                                train_y=y_train,
                                val_x=[class_x_val, word_x_val],
                                val_y=y_val)
        train_ppl = word_class_rnn_lm.evaluate_perplexity(x=[class_x_train,
                                                             word_x_train], y_true=y_train)
        test_ppl = word_class_rnn_lm.evaluate_perplexity(x=[class_x_test,
                                                            word_x_test], y_true=y_test)
        val_ppl = word_class_rnn_lm.evaluate_perplexity(x=[class_x_val,
                                                           word_x_val], y_true=y_val)
        label_probabilities = word_class_rnn_lm.predict(x=[class_x_train, word_x_train], true_y=y_train)
        print('Neural Perplexity: Train: {}, Test: {}, Val: {}'.format(round(train_ppl, 3),
                                                                       round(test_ppl, 3),
                                                                       round(val_ppl, 3)))

        # Converting to ARPA Format
        arpa_lm = ARPALM(word2idx=word_vocab)
        if args.input_ngram_lm:
            arpa_lm.read_lm(args.input_ngram_lm)
        else:
            arpa_lm.read_lm(args.output_base_dir + '/' + args.language + '/ngram/3gram_kn_interp.lm')
        arpa_lm.convert_to_ngram(label_probabilities)

        if args.output_ngram_lm:
            arpa_lm.write_arpa_format(args.output_ngram_lm)
        else:
            arpa_lm.write_arpa_format(save_dir + '/' + args.language + '_' + args.type + '_' + 'neural_3gram.lm')

        print('----------')
    else:
        model = RNNLM(type=args.type,
                      max_seq_len=max_sent_len,
                      max_word_len=max_word_len,
                      word_vocab_size=len(word_vocab),
                      char_vocab_size=len(char_vocab),
                      save_dir=save_dir,
                      config=config)
        label_probabilities = None
        train_ppl, test_ppl, val_ppl = None, None, None
        if args.type == 'word':
            model.train(train_x=[word_x_train], train_y=y_train, val_x=word_x_val, val_y=y_val)
            train_ppl = model.evaluate_perplexity(x=[word_x_train], y_true=y_train)
            test_ppl = model.evaluate_perplexity(x=[word_x_test], y_true=y_test)
            val_ppl = model.evaluate_perplexity(x=[word_x_val], y_true=y_val)
            label_probabilities = model.predict(x=[word_x_train], true_y=y_train)
        elif args.type == 'char':
            model.train(train_x=[char_x_train], train_y=y_train, val_x=char_x_val, val_y=y_val)
            train_ppl = model.evaluate_perplexity(x=[char_x_train], y_true=y_train)
            test_ppl = model.evaluate_perplexity(x=[char_x_test], y_true=y_test)
            val_ppl = model.evaluate_perplexity(x=[char_x_val], y_true=y_val)
            label_probabilities = model.predict(x=[char_x_train], true_y=y_train)
        elif args.type == 'word_char':
            model.train(train_x=[char_x_train, word_x_train], train_y=y_train,
                        val_x=[char_x_val, word_x_val],
                        val_y=y_val)
            train_ppl = model.evaluate_perplexity(x=[char_x_train, word_x_train], y_true=y_train)
            test_ppl = model.evaluate_perplexity(x=[char_x_test, word_x_test], y_true=y_test)
            val_ppl = model.evaluate_perplexity(x=[char_x_val, word_x_val], y_true=y_val)
            label_probabilities = model.predict(x=[char_x_train, word_x_train], true_y=y_train)

        print('Neural Perplexity: Train: {}, Test: {}, Val: {}'.format(round(train_ppl, 3),
                                                                       round(test_ppl, 3),
                                                                       round(val_ppl, 3)))

        # Converting to ARPA Format
        arpa_lm = ARPALM(word2idx=word_vocab)
        if args.input_ngram_lm:
            arpa_lm.read_lm(args.input_ngram_lm)
        else:
            arpa_lm.read_lm(args.output_base_dir + '/' + args.language + '/ngram/3gram_kn_interp.lm')
        arpa_lm.convert_to_ngram(label_probabilities)

        if args.output_ngram_lm:
            arpa_lm.write_arpa_format(args.output_ngram_lm)
        else:
            arpa_lm.write_arpa_format(save_dir + '/' + args.language + '_' + args.type + '_' + 'neural_3gram.lm')
        

if __name__ == '__main__':
    main()
