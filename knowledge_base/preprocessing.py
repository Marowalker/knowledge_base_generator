from knowledge_base.utils import *
import constants
import pickle


def make_pickle(infile, outfile):
    chem_id = make_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
    dis_id = make_vocab(constants.ENTITY_PATH + 'disease2id.txt')
    rel_id = make_vocab(constants.ENTITY_PATH + 'relation2id.txt')

    head, tail, rel, head_neg, tail_neg = process_triples(infile, chem_id, dis_id, rel_id)

    sequence_dict = {
        'head': head,
        'tail': tail,
        'rel': rel,
        'head_neg': head_neg,
        'tail_neg': tail_neg
    }
    with open(outfile, 'wb') as f:
        pickle.dump(sequence_dict, f)

    return sequence_dict


def wordnet_pickle(infile, outfile):
    entity_id = make_wordnet_vocab(constants.WORDNET_PATH + 'wordnet-entities.txt')
    relation_id = make_wordnet_vocab(constants.WORDNET_PATH + 'wordnet-relations.txt')

    head, tail, rel, head_neg, tail_neg = wordnet_triple(infile, entity_id, relation_id)

    sequence_dict = {
        'head': head,
        'tail': tail,
        'rel': rel,
        'head_neg': head_neg,
        'tail_neg': tail_neg
    }
    with open(outfile, 'wb') as f:
        pickle.dump(sequence_dict, f)

    return sequence_dict

