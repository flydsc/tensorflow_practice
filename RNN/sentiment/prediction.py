import tensorflow as tf
import load_test
import numpy as np

max_len, data_len, word_size, data= load_test.loadfile()

lr = 0.001
train_iters = 50000
batch_size = 128

n_inputs = max_len
n_hidden_units = 128

x = tf.placeholder(tf.int32, )
y = tf.placeholder(tf.int32, )

#define weights

weights = {
    "out": tf.Variable(tf.random_normal())
}

biases = {
    "out": tf.Variable(0.1, )
}

saver = tf.train.Saver()

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
