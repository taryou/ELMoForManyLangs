#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import glob
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from .modules.embedding_layer import EmbeddingLayer
from .dataloader import load_embedding
from .utils import dict2namedtuple
from collections import Counter
import numpy as np
from distutils.util import strtobool
from .frontend import create_batches, PackObj, TrainModel


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def divide(data, valid_size):
    valid_size = min(valid_size, len(data) // 10)
    random.shuffle(data)
    return data[valid_size:], data[:valid_size]


def break_sentence(sentence, max_sent_len):
    """
    For example, for a sentence with 70 words, supposing the the `max_sent_len'
    is 30, break it into 3 sentences.

    :param sentence: list[str] the sentence
    :param max_sent_len:
    :return:
    """
    ret = []
    cur = 0
    length = len(sentence)
    while cur < length:
        if cur + max_sent_len + 5 >= length:
            ret.append(sentence[cur: length])
            break
        ret.append(sentence[cur: min(length, cur + max_sent_len)])
        cur += max_sent_len
    return ret


def read_corpus(path, max_chars=None, max_sent_len=20):
    """
    read raw text file
    :param path: str
    :param max_chars: int
    :param max_sent_len: int
    :return:
    """

    if os.path.isfile(path):
        return read_corpus_original(path, max_chars, max_sent_len)
    elif os.path.isdir(path):
        threadlist = []
        datasets = []
        for f in glob.glob(os.path.join(path, '*')):
            thread = threading.Thread(target=read_corpus_original, args=([f, max_chars, max_sent_len, datasets]),
                                      name="thread-%s" % f)
            threadlist.append(thread)

        for th in threadlist:
            th.start()

        for th in threadlist:
            th.join()
        return datasets


def read_corpus_original(path, max_chars=None, max_sent_len=20, results=None):
    """
    read raw text file
    :param path: str
    :param max_chars: int
    :param max_sent_len: int
    :return:
    """
    data = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append('<bos>')
            for token in line.strip().split():
                if max_chars is not None and len(token) + 2 > max_chars:
                    token = token[:max_chars - 2]
                data.append(token)
            data.append('<eos>')
    dataset = break_sentence(data, max_sent_len)
    if results is not None:
        results.extend(dataset)
    return dataset


def eval_model(model, valid):
    model.eval()
    if model.config['classifier']['name'].lower() == 'cnn_softmax' or \
            model.config['classifier']['name'].lower() == 'sampled_softmax':
        model.classify_layer.update_embedding_matrix()
    total_loss, total_tag = 0.0, 0
    valid_w, valid_c, valid_lens, valid_masks = valid
    for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
        loss_forward, loss_backward = model.forward(w, c, masks)
        total_loss += loss_forward.sum().data.item()
        n_tags = sum(lens)
        total_tag += n_tags
    model.train()
    return np.exp(total_loss / total_tag)


def prarallel_reader(train_w, train_c, train_lens, train_masks, parallel):
    if parallel < 1:
        parallel = 1
    batch_w = []
    batch_c = []
    batch_l = []
    batch_m = []
    indices = torch.tensor(range(parallel))
    print(type(train_w[0]), type(train_c[0]), type(train_lens[0]), type(train_masks[0]))

    for i in range(len(train_w)):
        c = i + 1
        batch_w.append(train_w[i])
        batch_c.append(train_c[i])
        batch_l.extend(train_lens[i])
        batch_m.append(PackObj(train_masks[i]))
        if c % parallel == 0:
            try:
                yield torch.cat(batch_w, dim=0), torch.cat(batch_c, dim=0), batch_l, tuple(batch_m), indices
            except:
                pass
            batch_w = []
            batch_c = []
            batch_l = []
            batch_m = []


def average_gradients(model):
    size = float(dist.get_world_size())
    update = 0
    for param in model.parameters():
        if param.grad is None:
            data = param.data.new_tensor(param.data)
            data.fill_(0)
        else:
            data = param.grad.data
        dist.all_reduce(data, op=dist.reduce_op.SUM)
        update += torch.sum(torch.abs(data))
        if param.grad is not None:
            param.grad.data = data / size
    return update.item()


def train_model(epoch, opt, model, optimizer,
                train, valid, test, best_train, best_valid, test_result, rank, eps=1e-4):
    """
    Training model for one epoch

    :param epoch:
    :param opt:
    :param model:
    :param optimizer:
    :param train:
    :param best_train:
    :param valid:
    :param best_valid:
    :param test:
    :param test_result:
    :return:
    """
    model.train()

    total_loss, total_tag = 0.0, 0
    cnt = 0
    start_time = time.time()

    train_w, train_c, train_lens, train_masks = train

    lst = list(range(len(train_w)))
    random.shuffle(lst)

    train_w = [train_w[l] for l in lst]
    train_c = [train_c[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]
    train_masks = [train_masks[l] for l in lst]

    L = len(lst)

    #for w, c, lens, masks in zip(train_w, train_c, train_lens, train_masks):
    while True:
        if cnt < L:
            w = train_w[cnt]
            c = train_c[cnt]
            lens = train_lens[cnt]
            masks = train_masks[cnt]

            model.zero_grad()
            loss_forward, loss_backward = model.forward(w, c, masks)

            loss = (loss_forward + loss_backward) / 2.0
            total_loss += loss_forward.data.item()
            n_tags = sum(lens)
            total_tag += n_tags
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)

        update = average_gradients(model)
        optimizer.step()

        cnt += 1

        if rank == 0:
            if cnt * opt.batch_size % 1024 == 0:
                logging.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
                    epoch, cnt, optimizer.param_groups[0]['lr'],
                    np.exp(total_loss / total_tag), time.time() - start_time
                ))
                start_time = time.time()
            if cnt % opt.eval_steps == 0 or cnt % len(train_w) == 0:
                if valid is None:
                    train_ppl = np.exp(total_loss / total_tag)
                    logging.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f}".format(
                        epoch, cnt, optimizer.param_groups[0]['lr'], train_ppl))
                    if train_ppl < best_train:
                        best_train = train_ppl
                        logging.info("New record achieved on training dataset!")

                        model.save_model(opt.model, opt.save_classify_layer)
                else:
                    valid_ppl = eval_model(model, valid)
                    logging.info("Epoch={} iter={} lr={:.6f} valid_ppl={:.6f}".format(
                        epoch, cnt, optimizer.param_groups[0]['lr'], valid_ppl))

                    if valid_ppl < best_valid:
                        model.save_model(opt.model, opt.save_classify_layer)
                        best_valid = valid_ppl
                        logging.info("New record achieved!")

                        if test is not None:
                            test_result = eval_model(model, test)
                            logging.info("Epoch={} iter={} lr={:.6f} test_ppl={:.6f}".format(
                                epoch, cnt, optimizer.param_groups[0]['lr'], test_result))

        if update < eps:
            break

    return best_train, best_valid, test_result


def get_truncated_vocab(dataset, min_count):
    """

    :param dataset:
    :param min_count: int
    :return:
    """
    word_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)

    word_count = list(word_count.items())
    word_count.sort(key=lambda x: x[1], reverse=True)

    i = 0
    for word, count in word_count:
        if count < min_count:
            break
        i += 1

    logging.info('Truncated word count: {0}.'.format(sum([count for word, count in word_count[i:]])))
    logging.info('Original vocabulary size: {0}.'.format(len(word_count)))
    return word_count[:i]


def create_vocab():
    cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    cmd.add_argument('--parallel', default=True, type=strtobool, help='DataParallel training')
    cmd.add_argument('--train_path', required=True, help='The path to the training file.')
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument('--max_sent_len', type=int, default=20, help='maximum sentence length.')
    cmd.add_argument('--min_count', type=int, default=5, help='minimum word count.')
    cmd.add_argument('--max_vocab_size', type=int, default=150000, help='maximum vocabulary size.')

    cmd.add_argument('--config_path', required=True, help='the path to the config file.')
    cmd.add_argument("--word_embedding", help="The path to word vectors.")

    opt = cmd.parse_args(sys.argv[2:])

    with open(opt.config_path, 'r') as fin:
        config = json.load(fin)

    token_embedder_name = config['token_embedder']['name'].lower()
    token_embedder_max_chars = config['token_embedder'].get('max_characters_per_token', None)
    if token_embedder_name == 'cnn':
        train_data = read_corpus(opt.train_path, token_embedder_max_chars, opt.max_sent_len)
    elif token_embedder_name == 'lstm':
        train_data = read_corpus(opt.train_path, opt.max_sent_len)
    else:
        raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))

    # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
    vocab = get_truncated_vocab(train_data, opt.min_count)

    if opt.word_embedding is not None:
        embs = load_embedding(opt.word_embedding)
        word_lexicon = {word: i for i, word in enumerate(embs[0])}
    else:
        embs = None
        word_lexicon = {}

    # Ensure index of '<oov>' is 0
    for special_word in ['<oov>', '<bos>', '<eos>', '<pad>']:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)

    for word, _ in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)

    # Character Lexicon
    if config['token_embedder']['char_dim'] > 0:
        char_lexicon = {}
        for sentence in train_data:
            for word in sentence:
                for ch in word:
                    if ch not in char_lexicon:
                        char_lexicon[ch] = len(char_lexicon)

        for special_char in ['<bos>', '<eos>', '<oov>', '<pad>', '<bow>', '<eow>']:
            if special_char not in char_lexicon:
                char_lexicon[special_char] = len(char_lexicon)

        char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False)
        logging.info('Char embedding size: {0}'.format(len(char_emb_layer.word2id)))
    else:
        char_lexicon = None
        char_emb_layer = None

    with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    if config['token_embedder']['char_dim'] > 0:
        with codecs.open(os.path.join(opt.model, 'char.dic'), 'w', encoding='utf-8') as fpo:
            for ch, i in char_emb_layer.word2id.items():
                print('{0}\t{1}'.format(ch, i), file=fpo)


def train():
    cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=1, type=int, help='The random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')
    cmd.add_argument('--resume', default=False, type=strtobool, help='Resume training')
    cmd.add_argument('--rank', type=int, help='rank of distributed process', required=True)
    cmd.add_argument('--world_size', type=int, help='world_size of distributed process', required=True)
    cmd.add_argument('--init_method', type=str, help='init method of distributed process', required=True)

    cmd.add_argument('--train_path', required=True, help='The path to the training file.')
    cmd.add_argument('--valid_path', help='The path to the development file.')
    cmd.add_argument('--test_path', help='The path to the testing file.')

    cmd.add_argument('--config_path', required=True, help='the path to the config file.')
    cmd.add_argument("--word_embedding", help="The path to word vectors.")

    cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adagrad'],
                     help='the type of optimizer: valid options=[sgd, adam, adagrad]')
    cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')

    cmd.add_argument("--model", required=True, help="path to save model")

    cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')

    cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')

    cmd.add_argument('--max_sent_len', type=int, default=20, help='maximum sentence length.')

    cmd.add_argument('--save_classify_layer', default=False, action='store_true',
                     help="whether to save the classify layer")

    cmd.add_argument('--valid_size', type=int, default=0, help="size of validation dataset when there's no valid.")
    cmd.add_argument('--eval_steps', required=False, type=int, help='report every xx batches.')

    opt = cmd.parse_args(sys.argv[2:])

    # initialize distributed comp.
    dist.init_process_group(backend="nccl",
                            init_method=opt.init_method,
                            world_size=opt.world_size,
                            rank=opt.rank)

    with open(opt.config_path, 'r') as fin:
        config = json.load(fin)

    # Dump configurations
    print(opt)
    print(config)

    # set seed.
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        # torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    device = torch.device(opt.gpu) if opt.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')

    token_embedder_name = config['token_embedder']['name'].lower()
    token_embedder_max_chars = config['token_embedder'].get('max_characters_per_token', None)
    if token_embedder_name == 'cnn':
        train_data = read_corpus(opt.train_path, token_embedder_max_chars, opt.max_sent_len)
    elif token_embedder_name == 'lstm':
        train_data = read_corpus(opt.train_path, opt.max_sent_len)
    else:
        raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))

    logging.info('training instance: {}, training tokens: {}.'.format(len(train_data),
                                                                      sum([len(s) - 1 for s in train_data])))

    if opt.valid_path is not None:
        if token_embedder_name == 'cnn':
            valid_data = read_corpus(opt.valid_path, token_embedder_max_chars, opt.max_sent_len)
        elif token_embedder_name == 'lstm':
            valid_data = read_corpus(opt.valid_path, opt.max_sent_len)
        else:
            raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))
        logging.info('valid instance: {}, valid tokens: {}.'.format(len(valid_data),
                                                                    sum([len(s) - 1 for s in valid_data])))
    elif opt.valid_size > 0:
        train_data, valid_data = divide(train_data, opt.valid_size)
        logging.info('training instance: {}, training tokens after division: {}.'.format(
            len(train_data), sum([len(s) - 1 for s in train_data])))
        logging.info('valid instance: {}, valid tokens: {}.'.format(
            len(valid_data), sum([len(s) - 1 for s in valid_data])))
    else:
        valid_data = None

    if opt.test_path is not None:
        if token_embedder_name == 'cnn':
            test_data = read_corpus(opt.test_path, token_embedder_max_chars, opt.max_sent_len)
        elif token_embedder_name == 'lstm':
            test_data = read_corpus(opt.test_path, opt.max_sent_len)
        else:
            raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))
        logging.info('testing instance: {}, testing tokens: {}.'.format(
            len(test_data), sum([len(s) - 1 for s in test_data])))
    else:
        test_data = None

    if config['token_embedder']['char_dim'] > 0:
        char_lexicon = {}
        with codecs.open(os.path.join(opt.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                char_lexicon[token] = int(i)
        char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False)
        logging.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
    else:
        char_lexicon = None
        char_emb_layer = None

    if opt.word_embedding is not None:
        embs = load_embedding(opt.word_embedding)
    else:
        embs = None

    word_lexicon = {}
    with codecs.open(os.path.join(opt.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            word_lexicon[token] = int(i)

    # Word Embedding
    if config['token_embedder']['word_dim'] > 0:
        word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=embs)
        logging.info('Word embedding size: {0}'.format(len(word_emb_layer.word2id)))
    else:
        word_emb_layer = None
        logging.info('Vocabulary size: {0}'.format(len(word_lexicon)))

    train = create_batches(
        train_data, opt.batch_size, word_lexicon, char_lexicon, config, device=device)

    if opt.eval_steps is None:
        opt.eval_steps = len(train[0])
    logging.info('Evaluate every {0} batches.'.format(opt.eval_steps))

    if valid_data is not None:
        valid = create_batches(
            valid_data, opt.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, device=device)
    else:
        valid = None

    if test_data is not None:
        test = create_batches(
            test_data, opt.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, device=device)
    else:
        test = None

    label_to_ix = word_lexicon
    logging.info('vocab size: {0}'.format(len(label_to_ix)))

    nclasses = len(label_to_ix)

    model = TrainModel(config, word_emb_layer, char_emb_layer, nclasses, device)
    model.to(device)
    if os.path.exists(os.path.join(opt.model, 'token_embedder.pkl')):
        logging.info('use resume')
        model.load_model(opt.model)
    logging.info(str(model))

    need_grad = lambda x: x.requires_grad

    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
    elif opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)
    elif opt.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(filter(need_grad, model.parameters()), lr=opt.lr)
    else:
        raise ValueError('Unknown optimizer {}'.format(opt.optimizer.lower()))

    try:
        os.makedirs(opt.model)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

    best_train = 1e+8
    best_valid = 1e+8
    test_result = 1e+8

    for epoch in range(opt.max_epoch):
        best_train, best_valid, test_result = train_model(epoch, opt, model, optimizer,
                                                          train, valid, test, best_train, best_valid, test_result, opt.rank)
        if opt.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= opt.lr_decay

    if valid_data is None:
        logging.info("best train ppl: {:.6f}.".format(best_train))
    elif test_data is None:
        logging.info("best train ppl: {:.6f}, best valid ppl: {:.6f}.".format(best_train, best_valid))
    else:
        logging.info("best train ppl: {:.6f}, best valid ppl: {:.6f}, test ppl: {:.6f}.".format(best_train, best_valid,
                                                                                                test_result))


def test():
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument("--input", help="the path to the raw text file.")
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')

    args = cmd.parse_args(sys.argv[2:])

    device = torch.device(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')

    args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

    with open(args2.config_path, 'r') as fin:
        config = json.load(fin)

    if config['token_embedder']['char_dim'] > 0:
        char_lexicon = {}
        with codecs.open(os.path.join(args.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                char_lexicon[token] = int(i)
        char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False)
        logging.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
    else:
        char_lexicon = None
        char_emb_layer = None

    word_lexicon = {}
    with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            word_lexicon[token] = int(i)

    if config['token_embedder']['word_dim'] > 0:
        word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
        logging.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
    else:
        word_emb_layer = None

    model = TrainModel(config, word_emb_layer, char_emb_layer, len(word_lexicon), device)
    model = model.to(device)

    logging.info(str(model))
    model.load_model(args.model)
    if config['token_embedder']['name'].lower() == 'cnn':
        test = read_corpus(args.input, config['token_embedder']['max_characters_per_token'], max_sent_len=10000)
    elif config['token_embedder']['name'].lower() == 'lstm':
        test = read_corpus(args.input, max_sent_len=10000)
    else:
        raise ValueError('')

    test_w, test_c, test_lens, test_masks = create_batches(
        test, args.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, device=device)

    test_result = eval_model(model, (test_w, test_c, test_lens, test_masks))

    logging.info("test_ppl={:.6f}".format(test_result))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
