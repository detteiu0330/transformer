#train by online learning

import io
import progressbar
import argparse
import datetime
import random
import cloudpickle

import torch
from torch import nn as N
from torch import optim as O

from padding import padding

from encoder import Encoder
from decoder import Decoder


BOS = 0
EOS = 1
UNK = 2
PAD = 3


def count_lines(path):
    with io.open(path, encoding='utf-8') as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with io.open(path, encoding='utf-8') as f:
        # +4 for BOS, EOS, UNK and PAD
        word_ids = {line.strip(): i + 4 for i, line in enumerate(f)}
    word_ids['<BOS>'] = 0
    word_ids['<EOS>'] = 1
    word_ids['<UNK>'] = 2
    word_ids['<PAD>'] = 3
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar(maxval=n_lines)
    data = []
    print('loading...: %s' % path)
    with io.open(path, encoding='utf-8') as f:
        for line in bar(f):
            words = line.strip().split()
            array = torch.LongTensor([vocabulary.get(w, UNK) for w in words])
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = 0
    for s in data:
        for w in s:
            if w == UNK:
                unknown += 1
    total = sum(s.size(0) for s in data)
    return unknown / total


def train():
    parser = argparse.ArgumentParser(description='Transformer by Pytorch')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source', help='source sentence list for validation')
    parser.add_argument('--validation-target', help='target sentence list for validation')
    parser.add_argument('--min-source-sentence', type=int, default=1, help='minimum length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=20, help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1, help='minimum length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=20, help='maximum length of target sentence')
    parser.add_argument('--batch_size', type=int, default=6, help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of sweeps over the dataset to train')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--embed_size', type=int, default=4, help='embed size')
    parser.add_argument('--hidden_size', type=int, default=4, help='hidden size')
    parser.add_argument('--device', default="cuda", help='number of using device')
    args = parser.parse_args()

    # load train data set
    print('[{}] Loading data set... (this may take several minutes)'.format(datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    source_vocab_size = len(source_ids)
    target_vocab_size = len(target_ids)
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)
    train_source_unknown = calculate_unknown_ratio(train_source)
    train_target_unknown = calculate_unknown_ratio(train_target)
    assert len(train_source) == len(train_target), "source data size and target data size don't accord."
    print('[{}] Data set loaded.'.format(datetime.datetime.now()))
    # print information of train data
    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_source))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    # make id to source dictionary
    source_words = {i: w for w, i in source_ids.items()}
    target_words = {i: w for w, i in target_ids.items()}

    # assign vectors of bos and eos
    bos = torch.zeros(1, dtype=torch.int64, device=args.device)
    eos = torch.ones(1, dtype=torch.int64, device=args.device)

    # make instances of encoder and decoder
    encoder = Encoder(source_vocab_size, args.hidden_size).to(device=args.device)
    decoder = Decoder(target_vocab_size, args.hidden_size, device=args.device).to(device=args.device)

    # define optimizer
    encoder_optimizer = O.Adam(encoder.parameters())
    decoder_optimizer = O.Adam(decoder.parameters())

    # shuffle train data set
    train_data = [(s, t) for s, t in zip(train_source, train_target)]
    random.shuffle(train_data)

    # define loss function
    criterion = N.CrossEntropyLoss(ignore_index=PAD)

    # start training
    for epoch in range(args.epoch):
        step = 0
        for (source, target) in train_data:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            decoder_target = torch.cat((target.to(args.device), eos), dim=0)
            decoder_source = padding(target, args.max_source_sentence)
            decoder_target = padding(decoder_target, args.max_target_sentence+1)
            encoder_output = encoder(source.to(args.device))
            output_array = decoder(decoder_source.to(args.device), encoder_output.to(args.device))
            loss = criterion(output_array, decoder_target.to(args.device))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print("epoch: {},step: {}, loss = {}".format(epoch, step, float(loss)))
            step += 1
        with open('weights/encoder.pkl', 'wb') as f:
            cloudpickle.dump(encoder, f)
        with open('weights/decoder.pkl', 'wb') as f:
            cloudpickle.dump(decoder, f)


if __name__ == '__main__':
    train()






