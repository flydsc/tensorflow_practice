import config
import tensorflow as tf
from tensorflow.contrib import seq2seq, rnn
import data_loader


configs = config.para_config()
data = data_loader.Data()


class Model():
    def __init__(self):
        self.input_data = tf.placeholder(tf.int32, [configs.batch_size, configs.seq_length])
        self.target_data = tf.placeholder(tf.int32, [configs.batch_size, configs.seq_length])
        self.lr = tf.placeholder(tf.float32, [])

        #cell definition
        self.cell = rnn.BasicLSTMCell(configs.state_size)
        self.cell = rnn.MultiRNNCell([self.cell] * configs.num_layers)
        self.init_state = self.cell.zero_state(configs.batch_size, tf.float32)

        # para definitions
        w = tf.get_variable('softmax_w', [configs.state_size, data.vocab_size])
        b = tf.get_variable('softmax_b', [data.vocab_size])

        #embedding
        embedding = tf.get_variable('embedding', [data.vocab_size, configs.state_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        #output
        output, last_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.init_state)
        output = tf.reshape(output, [-1, configs.state_size])

        #logit computation
        self.logits = tf.matmul(output, w) + b
        self.probs = tf.nn.softmax(self.logits)
        self.last_state = last_state

        #comparison
        target = tf.reshape(self.target_data, [-1])
        loss = seq2seq.sequence_loss([self.logits], [target], [tf.ones_like(target, dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / configs.batch_size

        #optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, configs.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

