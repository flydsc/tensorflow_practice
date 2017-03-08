import tensorflow as tf
import numpy as np


def add_layer(input_data, in_dim, out_dim, act_fun=None):
    Weight = tf.Variable(tf.random_uniform([in_dim, out_dim]))
    bias = tf.Variable(tf.zeros([1, out_dim]) + 0.1)
    Wx_plus_b = tf.matmul(input_data, Weight) + bias
    if act_fun:
        output = act_fun(Wx_plus_b)
    else:
        output = Wx_plus_b
    return output

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer1 = add_layer(xs, 1, 10, act_fun=tf.nn.relu)
prediction = add_layer(layer1, 10, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs: x_data, ys:y_data})
        if i % 50 == 0:
            print sess.run(loss, feed_dict={xs: x_data, ys: y_data})