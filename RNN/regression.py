import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class RNNLSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size])
        self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size])
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()
        self.compute_cost()


    def add_input_layer(self):
        in_x = tf.reshap(self.xs, [-1, self.input_size])
        w_in = self._weight_variable([self.input_size, self.cell_size])
        bias_in = self._bias_variable([self.cell_size,])
        in_x_mu = tf.matmul(in_x, w_in) + bias_in
        self.out_x_l = tf.reshape(in_x_mu, [-1, self.n_steps, self.cell_size])#batch , step, cell

    def add_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_output, self.final_state = tf.nn.dynamic_rnn(cell, self.out_x_l, initial_state=self.init_state, time_major=False)


    def compute_cost(self):
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                tf.cast(self.batch_size, tf.float32),
                name='average_cost')

    def ms_error(self, y_pre, y_target):
        return tf.square(tf.sub(y_pre, y_target))

    def add_output_layer(self):
        out_x = tf.reshape(self.cell_output, [-1, self.cell_size])
        w_out = self._weight_variable([self.cell_size, self.output_size])
        b_out = self._bias_variable([self.output_size,])
        self.pred = tf.matmul(out_x, w_out) + b_out

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)