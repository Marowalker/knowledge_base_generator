import constants
from knowledge_base.preprocessing import make_pickle, wordnet_pickle
import pickle
from knowledge_base.transE import TransEModel
from knowledge_base.utils import make_vocab
import tensorflow as tf


def main_knowledge_base():
    if constants.IS_REBUILD == 1:
        print('Build data...')
        # make_chemicals()
        # make_diseases()
        # make_relations()
        # make_triples()
        # get_train_files()
        train_dict = make_pickle(constants.ENTITY_PATH + 'train2id.txt', constants.PICKLE + 'train_triple_data.pkl')
        val_dict = make_pickle(constants.ENTITY_PATH + 'valid2id.txt', constants.PICKLE + 'val_triple_data.pkl')
        test_dict = make_pickle(constants.ENTITY_PATH + 'test2id.txt', constants.PICKLE + 'test_triple_data.pkl')
    else:
        print('Load data...')
        with open(constants.PICKLE + 'train_triple_data.pkl', 'rb') as f:
            train_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'val_triple_data.pkl', 'rb') as f:
            val_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'test_triple_data.pkl', 'rb') as f:
            test_dict = pickle.load(f)

    print("Train shape: ", len(train_dict['head']))
    print("Test shape: ", len(test_dict['head']))
    print("Validation shape: ", len(val_dict['head']))

    props = ['head', 'tail', 'rel', 'head_neg', 'tail_neg']

    for prop in props:
        train_dict[prop].extend(val_dict[prop])

    transe = TransEModel(model_name=constants.MODEL_NAMES.format('transe', constants.JOB_IDENTITY), batch_size=512,
                         epochs=constants.EPOCHS,
                         score=constants.SCORE)
    transe.build(train_dict, test_dict)
    transe.train(early_stopping=True, patience=constants.PATIENCE)


def main_wordnet():
    if constants.IS_REBUILD == 1:
        print('Build data...')
        train_dict = wordnet_pickle(constants.WORDNET_PATH + 'wordnet-train.txt', constants.PICKLE + 'wordnet_train.pkl')
        val_dict = wordnet_pickle(constants.WORDNET_PATH + 'wordnet-valid.txt', constants.PICKLE + 'wordnet_val.pkl')
        test_dict = wordnet_pickle(constants.WORDNET_PATH + 'wordnet-test.txt', constants.PICKLE + 'wordnet_test.pkl')
    else:
        print('Load data...')
        with open(constants.PICKLE + 'wordnet_train.pkl', 'rb') as f:
            train_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'wordnet_val.pkl', 'rb') as f:
            val_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'wordnet_test.pkl', 'rb') as f:
            test_dict = pickle.load(f)

    print("Train shape: ", len(train_dict['head']))
    print("Test shape: ", len(test_dict['head']))
    print("Validation shape: ", len(val_dict['head']))

    props = ['head', 'tail', 'rel', 'head_neg', 'tail_neg']

    for prop in props:
        train_dict[prop].extend(val_dict[prop])

    with tf.device('/device:GPU:0'):

        transe = WordnetTransE(model_path=constants.TRAINED_MODELS + 'wordnet/', batch_size=32, epochs=constants.EPOCHS,
                               score=constants.SCORE)
        transe.build(train_dict, test_dict)
        transe.train(early_stopping=True, patience=constants.PATIENCE)
        all_emb = transe.load('data/w2v_model/wordnet_embeddings.pkl')
        print(all_emb)
        # transe.load()


if __name__ == '__main__':
    main_knowledge_base()
    # main_wordnet()
