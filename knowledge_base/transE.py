import pickle

import tensorflow as tf
import constants
import os
import numpy as np
from data_utils import count_vocab, count_wordnet, Timer, Log
from sklearn.utils import shuffle


tf.compat.v1.disable_eager_execution()


class TransEModel:
    def __init__(self, model_name, batch_size, epochs, score):
        self.model_path = constants.TRAINED_MODELS
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model_name = model_name
        self.batch_size = batch_size
        self.chem_size = count_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
        self.dis_size = count_vocab(constants.ENTITY_PATH + 'disease2id.txt')
        self.rel_size = count_vocab(constants.ENTITY_PATH + 'relation2id.txt')
        self.epochs = epochs
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.dataset_train = None
        self.dataset_val = None
        self.score = score

    def score_function(self, h, t, r):
        if self.score == 'l1':
            score = tf.reduce_sum(input_tensor=tf.abs(h + r - t))
        else:
            score = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(h + r - t)))
        return score

    def _add_placeholder(self):
        self.head = tf.compat.v1.placeholder(name='head', shape=[None], dtype=tf.int32)
        self.tail = tf.compat.v1.placeholder(name='tail', shape=[None], dtype=tf.int32)
        self.rel = tf.compat.v1.placeholder(name='rel', shape=[None], dtype=tf.int32)
        self.head_neg = tf.compat.v1.placeholder(name='head_neg', shape=[None], dtype=tf.int32)
        self.tail_neg = tf.compat.v1.placeholder(name='tail_neg', shape=[None], dtype=tf.int32)

    def _add_embeddings(self):
        # generate embeddings
        with tf.compat.v1.variable_scope('embedding', reuse=tf.compat.v1.AUTO_REUSE):
            # embedding
            bound = 6 / tf.math.sqrt(float(constants.INPUT_W2V_DIM))
            with tf.compat.v1.variable_scope('embedding'):
                self.chemical_embedding = tf.compat.v1.get_variable(
                    name='chemical',
                    shape=[self.chem_size + 1, constants.INPUT_W2V_DIM],
                    initializer=tf.compat.v1.random_uniform_initializer(-bound, bound))
                self.chemical_embedding = tf.nn.l2_normalize(self.chemical_embedding, axis=1)

                self.disease_embedding = tf.compat.v1.get_variable(
                    name='disease',
                    shape=[self.dis_size + 1, constants.INPUT_W2V_DIM],
                    initializer=tf.compat.v1.random_uniform_initializer(-bound, bound))
                self.disease_embedding = tf.nn.l2_normalize(self.disease_embedding, axis=1)

                self.relation_embedding = tf.compat.v1.get_variable(
                    name='relation',
                    shape=[self.rel_size + 1, constants.INPUT_W2V_DIM],
                    initializer=tf.compat.v1.random_uniform_initializer(-bound, bound))
                self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, axis=1)
                tf.compat.v1.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

            with tf.compat.v1.name_scope('lookup'):
                self.h = tf.nn.embedding_lookup(params=self.chemical_embedding, ids=self.head)
                self.t = tf.nn.embedding_lookup(params=self.disease_embedding, ids=self.tail)
                self.r = tf.nn.embedding_lookup(params=self.relation_embedding, ids=self.rel)
                self.h_neg = tf.nn.embedding_lookup(params=self.chemical_embedding, ids=self.head_neg)
                self.t_neg = tf.nn.embedding_lookup(params=self.disease_embedding, ids=self.tail_neg)

    def _add_loss_op(self):
        score_pos = self.score_function(self.h, self.r, self.t)
        score_neg = self.score_function(self.h_neg, self.r, self.t_neg)
        self.predict = score_pos
        self.loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg),
                                  name='max_margin_loss')
        tf.compat.v1.summary.scalar(name=self.loss.op.name, tensor=self.loss)

    def _add_train_op(self):
        # self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.merge = tf.compat.v1.summary.merge_all()

    def build(self, data_train, data_val):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholder()
        self._load_data(data_train, data_val)
        self._add_embeddings()
        self._add_loss_op()
        self._add_train_op()

        timer.stop()

    def _load_data(self, train_data, val_data):
        self.dataset_train = train_data
        self.dataset_val = val_data

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            head = data['head'][start: start + self.batch_size]
            tail = data['tail'][start: start + self.batch_size]
            rel = data['rel'][start: start + self.batch_size]
            head_neg = data['head_neg'][start: start + self.batch_size]
            tail_neg = data['tail_neg'][start: start + self.batch_size]

            start += self.batch_size
            idx += 1
            yield head, tail, rel, head_neg, tail_neg

    def train(self, early_stopping=True, patience=10, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        best_loss = 100000
        n_epoch_no_improvement = 0

        saver = tf.compat.v1.train.Saver(max_to_keep=2)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            num_batch_train = len(self.dataset_train['head']) // self.batch_size + 1
            for e in range(self.epochs):
                print("\nStart of epoch %d" % (e + 1,))

                head_shuffled, tail_shuffled, rel_shuffled, head_neg_shuffled, tail_neg_shuffled = shuffle(
                    self.dataset_train['head'],
                    self.dataset_train['tail'],
                    self.dataset_train['rel'],
                    self.dataset_train['head_neg'],
                    self.dataset_train['tail_neg']
                )

                data = {
                    'head': head_shuffled,
                    'tail': tail_shuffled,
                    'rel': rel_shuffled,
                    'head_neg': head_neg_shuffled,
                    'tail_neg': tail_neg_shuffled
                }

                # Iterate over the batches of the dataset.
                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                    head, tail, rel, head_neg, tail_neg = batch

                    feed_dict = {
                        self.head: head,
                        self.tail: tail,
                        self.rel: rel,
                        self.head_neg: head_neg,
                        self.tail_neg: tail_neg
                    }

                    _, loss_train = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if idx % 1000 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

                if early_stopping:
                    num_batch_val = len(self.dataset_val['head']) // self.batch_size + 1

                    total_loss = []

                    data = {
                        'head': self.dataset_val['head'],
                        'tail': self.dataset_val['tail'],
                        'rel': self.dataset_val['rel'],
                        'head_neg': self.dataset_val['head_neg'],
                        'tail_neg': self.dataset_val['tail_neg']
                    }

                    for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                        head, tail, rel, head_neg, tail_neg = batch
                        loss = sess.run(self.loss, feed_dict={
                            self.head: head,
                            self.tail: tail,
                            self.rel: rel,
                            self.head_neg: head_neg,
                            self.tail_neg: tail_neg
                        })
                        total_loss.append(loss)

                    val_loss = np.mean(total_loss)
                    Log.log("Loss at epoch number {}: {}".format(e + 1, val_loss))
                    print("Previous best loss: ", best_loss)

                    if val_loss < best_loss:
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        saver.save(sess, self.model_name)
                        best_loss = val_loss
                        n_epoch_no_improvement = 0

                    else:
                        n_epoch_no_improvement += 1
                        Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                        if n_epoch_no_improvement >= patience:
                            print("Best loss: {}".format(best_loss))
                            break

                if not early_stopping:
                    saver.save(sess, self.model_name)

    def load_embedding(self, embedding_type='chemical'):
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            Log.log("Loading embedding type: {}".format(embedding_type))
            saver.restore(sess, self.model_name)

            if embedding_type == 'chemical':
                embeddings = sess.run(self.chemical_embedding)
            elif embedding_type == 'disease':
                embeddings = sess.run(self.disease_embedding)
            else:
                embeddings = sess.run(self.relation_embedding)
            return embeddings

# class WordnetTransE:
#     def __init__(self, model_path, batch_size, epochs, score):
#         self.model_path = model_path
#         if not os.path.exists(self.model_path):
#             os.makedirs(self.model_path)
#         self.batch_size = batch_size
#         self.ent_size = count_wordnet(constants.WORDNET_PATH + 'wordnet-entities.txt')
#         self.rel_size = count_wordnet(constants.WORDNET_PATH + 'wordnet-relations.txt')
#         self.epochs = epochs
#         self.initializer = tf.keras.initializers.GlorotUniform()
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
#         self.dataset_train = None
#         self.dataset_val = None
#         self.score = score
#
#     def _add_inputs(self):
#         self.head = tf.keras.Input(name='head', shape=(None,), dtype=tf.int32)
#         self.tail = tf.keras.Input(name='tail', shape=(None,), dtype=tf.int32)
#         self.rel = tf.keras.Input(name='rel', shape=(None,), dtype=tf.int32)
#         self.head_neg = tf.keras.Input(name='head_neg', shape=(None,), dtype=tf.int32)
#         self.tail_neg = tf.keras.Input(name='tail_neg', shape=(None,), dtype=tf.int32)
#
#     def _add_embeddings(self):
#         # generate embeddings
#         self.entity_embeddings = tf.Variable(self.initializer(shape=[self.ent_size + 1, 17],
#                                                               dtype=tf.float32), name='entity')
#         self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, axis=1)
#         self.relation_embeddings = tf.Variable(self.initializer(shape=[self.rel_size + 1, 17]),
#                                                dtype=tf.float32, name='relation')
#         self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)
#         # lookup embedding for scoring function
#         entity_lookup = LookupLayer(self.entity_embeddings, 'entity')
#         relation_lookup = LookupLayer(self.relation_embeddings, 'relation')
#         self.head_lookup = entity_lookup(self.head)
#         self.tail_lookup = entity_lookup(self.tail)
#         self.rel_lookup = relation_lookup(self.rel)
#         self.head_neg_lookup = entity_lookup(self.head_neg)
#         self.tail_neg_lookup = entity_lookup(self.tail_neg)
#
#     def score_function(self, h, t, r):
#         if self.score == 'l1':
#             score = tf.reduce_sum(input_tensor=tf.abs(h + r - t))
#         else:
#             score = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(h + r - t)))
#         return score
#
#     def _add_model(self):
#         self.model = tf.keras.Model(inputs=[self.head, self.tail, self.rel, self.head_neg, self.tail_neg],
#                                     outputs=[self.head_lookup, self.tail_lookup, self.rel_lookup, self.head_neg_lookup,
#                                              self.tail_neg_lookup])
#
#     def build(self, data_train, data_val):
#         timer = Timer()
#         timer.start("Building model...")
#
#         self._add_inputs()
#         self._load_data(data_train, data_val)
#         self._add_embeddings()
#         self._add_model()
#
#         timer.stop()
#
#     def _load_data(self, train_data, val_data):
#         self.dataset_train = train_data
#         self.dataset_val = val_data
#
#     def train(self, early_stopping=True, patience=10):
#         best_loss = 100000
#         n_epoch_no_improvement = 0
#
#         for e in range(self.epochs):
#             print("\nStart of epoch %d" % (e + 1,))
#
#             train_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_train)
#             train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
#
#             # Iterate over the batches of the dataset.
#             for idx, batch in enumerate(train_dataset):
#                 with tf.GradientTape() as tape:
#                     logits = self.model(batch, training=True)
#                     h, t, r, h_n, t_n = logits
#                     score_pos = self.score_function(h, t, r)
#                     score_neg = self.score_function(h_n, t_n, r)
#                     loss_value = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
#                     grads = tape.gradient(loss_value, self.model.trainable_weights)
#                 self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
#                 if idx % 100 == 0:
#                     Log.log("Iter {}, Loss: {} ".format(idx, loss_value))
#
#             if early_stopping:
#                 total_loss = []
#
#                 val_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_val)
#                 val_dataset = val_dataset.batch(self.batch_size)
#
#                 for idx, batch in enumerate(val_dataset):
#                     val_logits = self.model(batch, training=False)
#                     h, t, r, h_n, t_n = val_logits
#                     score_pos = self.score_function(h, t, r)
#                     score_neg = self.score_function(h_n, t_n, r)
#                     v_loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
#                     total_loss.append(float(v_loss))
#
#                 val_loss = np.mean(total_loss)
#                 Log.log("Loss at epoch number {}: {}".format(e + 1, val_loss))
#                 print("Previous best loss: ", best_loss)
#
#                 if val_loss < best_loss:
#                     Log.log('Save the model at epoch {}'.format(e + 1))
#                     self.model.save_weights(self.model_path)
#                     best_loss = val_loss
#                     n_epoch_no_improvement = 0
#
#                 else:
#                     n_epoch_no_improvement += 1
#                     Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
#                     if n_epoch_no_improvement >= patience:
#                         print("Best loss: {}".format(best_loss))
#                         break
#
#         if not early_stopping:
#             self.model.save_weights(self.model_path)
#
#     def save_model(self):
#         self.model.save_weights(self.model_path)
#
#     def load(self, load_file=None):
#         if not os.path.exists(load_file):
#             self.model.load_weights(self.model_path)
#             all_weights = []
#             for layer in self.model.layers:
#                 for weight in layer.weights:
#                     all_weights.append(weight)
#
#             new_weights = []
#             for w in all_weights:
#                 if 'entity' in w.name:
#                     w = self.entity_embeddings * w
#                 else:
#                     pass
#                     # w = self.relation_embeddings * w
#                 new_weights.append(w)
#
#             all_embeddings = tf.concat(new_weights, axis=0).numpy()
#             f = open(load_file, 'wb')
#             pickle.dump(all_embeddings, f)
#         else:
#             f = open(load_file, 'rb')
#             all_embeddings = pickle.load(f)
#         return all_embeddings


