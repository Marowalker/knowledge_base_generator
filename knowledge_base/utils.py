from collections import defaultdict
import random

import constants


def make_vocab(infile):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    # raw_vocab = defaultdict()
    id_vocab = defaultdict()
    for idx, line in enumerate(lines):
        if idx != 0:
            pairs = line.split('\t')
            name, name_id = pairs[0], pairs[1]
            # raw_vocab[name_id] = name
            id_vocab[name_id] = idx
    return id_vocab


def make_wordnet_vocab(infile):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    # raw_vocab = defaultdict()
    id_vocab = defaultdict()
    for idx, line in enumerate(lines):
        pairs = line.split()
        pairs = [p.strip() for p in pairs]
        name_id, name = pairs[0], pairs[1]
        # raw_vocab[name_id] = name
        id_vocab[name_id] = idx
    return id_vocab


def wordnet_triple_vocab():
    data = ['train', 'valid', 'test']
    res = defaultdict()
    for d in data:
        vocab = open(constants.WORDNET_PATH + 'wordnet-' + d + '.txt')
        lines = vocab.readlines()
        for line in lines:
            elems = line.strip().split()
            rel = elems[-1]
            pair = '_'.join([elems[0], elems[1]])
            res[pair] = rel
    return res


def process_triples(infile, chem_vocab, dis_vocab, rel_vocab):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    chems = list(chem_vocab.keys())
    dises = list(dis_vocab.keys())

    heads = []
    tails = []
    rels = []
    heads_neg = []
    tails_neg = []

    for idx, line in enumerate(lines):
        if idx != 0:
            triple = line.split('\t')
            head, tail, rel = triple[0], triple[1], triple[2]
            head_id = chem_vocab[head]
            tail_id = dis_vocab[tail]
            rel_id = rel_vocab[rel]
            threshold = random.random()
            if threshold < 0.5:
                head_neg = random.choice(chems)
                head_neg_id = chem_vocab[head_neg]
                tail_neg_id = tail_id
            else:
                tail_neg = random.choice(dises)
                tail_neg_id = dis_vocab[tail_neg]
                head_neg_id = head_id
            heads.append(head_id)
            tails.append(tail_id)
            rels.append(rel_id)
            heads_neg.append(head_neg_id)
            tails_neg.append(tail_neg_id)

    return heads, tails, rels, heads_neg, tails_neg


def wordnet_triple(infile, ent_vocab, rel_vocab):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    ents = list(ent_vocab.keys())

    heads = []
    tails = []
    rels = []
    heads_neg = []
    tails_neg = []

    for idx, line in enumerate(lines):
        if idx != 0:
            triple = line.split('\t')
            head, tail, rel = triple[0], triple[1], triple[2]
            head_id = ent_vocab[head]
            tail_id = ent_vocab[tail]
            rel_id = rel_vocab[rel]
            threshold = random.random()
            if threshold < 0.5:
                head_neg = random.choice(ents)
                head_neg_id = ent_vocab[head_neg]
                tail_neg_id = tail_id
            else:
                tail_neg = random.choice(ents)
                tail_neg_id = ent_vocab[tail_neg]
                head_neg_id = head_id
            heads.append(head_id)
            tails.append(tail_id)
            rels.append(rel_id)
            heads_neg.append(head_neg_id)
            tails_neg.append(tail_neg_id)

    return heads, tails, rels, heads_neg, tails_neg


