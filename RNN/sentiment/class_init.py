import tensorflow as tf
import load
import numpy as np
import load_test
# number 1 to 10 data
labels_n, max_len, data_len, word_size, data, Y = load.loadfile()
test_data = load_test.loadfile()
#para
lr = 0.001
train_iters = 50000
batch_size = 128

n_inputs = max_len
n_hidden_units = 128
n_classes = labels_n

x = tf.placeholder(tf.int32, [batch_size, n_inputs])
y = tf.placeholder(tf.int32, [batch_size, n_classes])

#define weights

weights = {
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    "out": tf.Variable(0.1, [n_classes,])
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    embedding = tf.get_variable('embedding', [word_size, n_hidden_units])
    inputs = tf.nn.embedding_lookup(embedding, X)
    # x_in = tf.matmul(inputs, weights['in']) + biases['in']

    # cell
    ##########################################
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state, time_major=False)



    # hidden layer for output as the final results
    #############################################
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

def RNNN(X, weights, biases):
    batch_size = 1
    # hidden layer for input to cell
    ########################################
    embedding = tf.get_variable('embedding', [word_size, n_hidden_units])
    inputs = tf.nn.embedding_lookup(embedding, X)
    # x_in = tf.matmul(inputs, weights['in']) + biases['in']

    # cell
    ##########################################
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state, time_major=False)



    # hidden layer for output as the final results
    #############################################
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
predd = RNNN(x, weights, biases)
prediction = tf.argmax(predd, 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    result = []
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    for ij in range(train_iters):
        if step * batch_size <= data_len:
            batch_xs = data[(step-1) * batch_size : step * batch_size]
        # print np.array(batch_xs).shape
            batch_ys = Y[(step-1) * batch_size : step * batch_size]
        else:
            remain = data_len - (step-1) * batch_size
            start = batch_size - remain
            batch_xs = data[(step-1) * batch_size :] + data[:start]
            batch_ys = Y[(step-1) * batch_size :] + Y[:start]
            step = 0
        b_y = np.array(batch_ys)
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: b_y,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: b_y,
        }))
        step += 1
    save_path = saver.save(sess, "./model.ckpt")
    print "Model saved in file: ", save_path
    for t in test_data:
        result.append(
            sess.run([prediction], feed_dict={
                x:t
            })
        )
    with open('result.txt', 'w') as out:
        for i in result:
            out.write(str(i) + '\n')


