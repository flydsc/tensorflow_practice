import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('../mnist_basic/MNIST_data', one_hot=True)

#para
lr = 0.001
train_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_inputs, n_steps])
y = tf.placeholder(tf.float32, [None, n_classes])

#define weights

weights = {
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    "in": tf.Variable(0.1, [n_hidden_units, ]),
    "out": tf.Variable(0.1, [n_classes,])
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    x = tf.reshape(X, [-1, n_inputs])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in, [-1, n_inputs, n_hidden_units])

    # cell
    ##########################################
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)



    # hidden layer for output as the final results
    #############################################
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < train_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1
