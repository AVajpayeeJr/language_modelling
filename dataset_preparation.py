import argparse
import codecs
from collections import defaultdict
import logging
import os
import shutil
import subprocess
import yaml

__author__ = 'avijitv@uw.edu'


def split_text(data_file_path, config, handle_oov=True):
    """ Reads source text and splits into train / test / val sets.

    Reads a given source file containing comma separated sentences in each line and splits into train / test / val lists
    of size described in the config dict. Only considers sentences of more than threshold tokens.

    Args:
        data_file_path (str): Source file path containing monolingual text (one space separated sentence per line)
        config (dict): Configuration dict for dataset construction hyper-parameters.
        handle_oov (bool): If True, restrict vocab based on frequency words and desired vocab size. Replace OOV in
            train / test / val with <unk>
            (default True)
    Returns:
        vocab (dict)
        train (:obj:list of :obj:str)
        test (:obj:list of :obj:str)
        val (:obj:list of :obj:str)
    """
    vocab = {}
    train, test, val = [], [], []
    train_sent_cnt = int(config['num_train_sentences'])
    test_sent_cnt = int(config['num_test_sentences'])
    val_sent_cnt = int(config['num_valid_sentences'])
    curr_sent_cnt = 0

    with codecs.open(data_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line:
                curr_sent_cnt += 1
                tokens = line.split()
                if curr_sent_cnt <=  val_sent_cnt:
                    val.append(tokens)
                elif curr_sent_cnt <= val_sent_cnt + test_sent_cnt:
                    test.append(tokens)
                elif curr_sent_cnt <= train_sent_cnt + val_sent_cnt + test_sent_cnt:
                    train.append(tokens)
                    if len(tokens) >= config['min_sentence_len']:
                        for t in tokens:
                            vocab[t] = vocab.get(t, 0) + 1
                elif curr_sent_cnt > train_sent_cnt + val_sent_cnt + test_sent_cnt:
                    break
    logging.debug('Number {} sentences: {}'.format('Train', len(train)))
    logging.debug('Number {} sentences: {}'.format('Test', len(val)))
    logging.debug('Number {} sentences: {}'.format('Valid', len(test)))

    if handle_oov:
        vocab, train, test, val = _handle_oov(config=config, orig_vocab=vocab, train=train, test=test, val=val)

    return vocab, train, test, val


def _handle_oov(config, orig_vocab, train, test, val):
    """ Restricts vocab and replaces OOV with <unk> token.

    Takes in an original vocab and restricts it first on threshold word frequency and (if size > desired) takes
    top n most frequent tokens. In-place replaces OOV tokens in train / test / val with <unk>.
    """
    logging.debug('Original Train vocab size: {}'.format(len(orig_vocab)))
    restricted_vocab = {k:v for k,v in orig_vocab.items() if v > config['word_min_count']}
    logging.debug('Restricted Train vocab size: {}'.format(len(restricted_vocab)))

    final_vocab_items = sorted(restricted_vocab.items(), key=lambda x: x[1], reverse=True)[:config['vocab_size']]
    final_vocab = dict(final_vocab_items)
    logging.debug('Final vocab size: {}'.format(len(final_vocab)))

    train_oov_replaced, train_total_tokens = 0, 0
    for i, sent in enumerate(train):
        for j, token in enumerate(sent):
            if token not in final_vocab:
                train[i][j] = '<unk>'
                train_oov_replaced += 1
            train_total_tokens += 1

    test_oov_replaced, test_total_tokens = 0, 0
    for i, sent in enumerate(test):
        for j, token in enumerate(sent):
            if token not in final_vocab:
                test[i][j] = '<unk>'
                test_oov_replaced += 1
            test_total_tokens += 1

    val_oov_replaced, val_total_tokens = 0, 0
    for i, sent in enumerate(val):
        for j, token in enumerate(sent):
            if token not in final_vocab:
                val[i][j] = '<unk>'
                val_oov_replaced += 1
            val_total_tokens += 1

    logging.debug('{}: OOV Replaced={} ({}/{})'.format('Train',
                                                       round(float(train_oov_replaced / train_total_tokens), 3),
                                                       train_oov_replaced,
                                                       train_total_tokens))
    logging.debug('{}: OOV Replaced={} ({}/{})'.format('Test',
                                                       round(float(test_oov_replaced / test_total_tokens), 3),
                                                       test_oov_replaced,
                                                       test_total_tokens))
    logging.debug('{}: OOV Replaced={} ({}/{})'.format('Valid',
                                                       round(float(val_oov_replaced / val_total_tokens), 3),
                                                       val_oov_replaced,
                                                       val_total_tokens))
    return final_vocab, train, test, val


def write_to_file(language_code, final_vocab, train_sentences, test_sentences, val_sentences):
    """ Writes vocab / train / test / val to a directory with given language code.
    """
    if not os.path.exists(language_code + '/'):
        os.makedirs(language_code)

    with codecs.open(language_code + '/vocab.txt', 'w', encoding='utf-8') as vocab_file:
        for i in final_vocab:
            vocab_file.write(i + '\n')

    with codecs.open(language_code + '/train.txt', 'w', encoding='utf-8') as train_file:
        for i in train_sentences:
            train_file.write(' '.join(i) + '\n')

    with codecs.open(language_code + '/test.txt', 'w', encoding='utf-8') as test_file:
        for i in test_sentences:
            test_file.write(' '.join(i) + '\n')

    with codecs.open(language_code + '/valid.txt', 'w', encoding='utf-8') as valid_file:
        for i in val_sentences:
            valid_file.write(' '.join(i) + '\n')


def write_brown_clusters(language_code, num_clusters, clustering_home_path):
    """ Runs Brown Clustering and writes clusters to file.

    Runs Brown Clustering through threaded-implementation (https://github.com/ajaech/brown-cluster) on train file and
    writes processed clusters and paths to directory containing vocab / train / test / val.
    """
    run_args = subprocess.run(args=[clustering_home_path + '/' + 'wcluster',
                                    '--c', str(num_clusters),
                                    '--text', str(language_code) + '/' + 'train.txt'],
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        run_args.check_returncode()
        output_dir = run_args.stdout.decode('utf-8').strip().split()[-1].split('/')[0]
        word2cluster_path, cluster_path2word = {}, defaultdict(list)
        with codecs.open(output_dir + '/' + 'paths') as paths_file:
            for line in paths_file:
                line = line.strip()
                if line:
                    cluster_path = line.split()[0]
                    word = line.split()[1]
                    word2cluster_path[word] = cluster_path
                    cluster_path2word[cluster_path].append(word)

        with codecs.open(str(language_code) + '/' + str(num_clusters) + '_clusters.txt', 'w') as out_file:
            for cluster_path in sorted(cluster_path2word.keys()):
                logging.debug('Cluster {}: {} words'.format(cluster_path, len(cluster_path2word[cluster_path])))
                for word in cluster_path2word[cluster_path]:
                    out_file.write('{}\t{}\n'.format(cluster_path, word))

        os.remove(str(language_code) + '/train.txt.int')
        os.remove(str(language_code) + '/train.txt.strdb')
        shutil.move(output_dir + '/paths', str(language_code) + '/' + str(num_clusters) + '_cluster_paths')
        shutil.rmtree(output_dir)
    except subprocess.CalledProcessError as err:
        err.message = err.message + '\nBrown Clustering failed.'
        raise err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_monolingual_file', help='Source monolingual text')
    parser.add_argument('--language', help='Language Code (Directory for processed data)')
    parser.add_argument('--cluster', action='store_true', default=True, help='Run Brown Clustering')
    parser.add_argument('--clustering_home_path', default='brown-cluster-master',
                        help='Path to Brown Clustering Directory')
    parser.add_argument('--num_clusters', type=int, default=300, help='Number of clusters when running Brown Clustering')
    parser.add_argument('--debug', action='store_true', default=True, help='Run with DEBUG logging level')
    parser.add_argument('--config', help='Config File')
    args = parser.parse_args()


    with open(args.config, 'r') as infile:
        config = yaml.load(infile)

    if args.debug:
        logging.basicConfig(format='%(levelname)s:%(funcName)s:%(lineno)s:\t%(message)s', level=logging.DEBUG)

    vocab, train_sentences, test_sentences, val_sentences = split_text(args.source_monolingual_file,
                                                                       config['data'])

    write_to_file(language_code=args.language,
                  final_vocab=vocab,
                  train_sentences=train_sentences,
                  test_sentences=test_sentences,
                  val_sentences=val_sentences)

    if args.cluster:
        write_brown_clusters(language_code=args.language, num_clusters=args.num_clusters,
                             clustering_home_path=args.clustering_home_path)


if __name__ == '__main__':
    main()
