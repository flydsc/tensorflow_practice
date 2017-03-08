from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def add_layer(input_data, in_dim, out_dim, act_fun=None):
    Weight = tf.Variable(tf.random_uniform([in_dim, out_dim]))
    bias = tf.Variable(tf.zeros([1, out_dim]) + 0.1)
    Wx_plus_b = tf.matmul(input_data, Weight) + bias
    if act_fun:
        output = act_fun(Wx_plus_b)
    else:
        output = Wx_plus_b
    return output


def compute_accuracy(v_x, v_y):
    global prediction
    pre_y = sess.run(prediction, feed_dict={xs: v_x})
    correct = tf.equal(tf.argmax(v_y, 1), tf.argmax(pre_y, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    result = sess.run(acc, feed_dict={xs: v_x, ys: v_y})
    return result

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, act_fun=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: x_batch, ys: y_batch})
        if i % 50 == 0:
            print compute_accuracy(mnist.test.images, mnist.test.labels)

