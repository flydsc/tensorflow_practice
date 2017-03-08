import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 12 + 88


# variable
weight = tf.Variable(tf.random_uniform([1], -100, 100))
bias = tf.Variable(tf.random_uniform([1], -100, 100))
#define formula
y = x_data * weight + bias

#tf structure init
loss = tf.reduce_mean(tf.square(y - y_data))
opt = tf.train.GradientDescentOptimizer(0.5)
train = opt.minimize(loss)
init = tf.global_variables_initializer()

### tf init
sess = tf.Session()
sess.run(init)




### start train
for i in range(201):
    sess.run(train)
    if i % 20 == 0:
        print i, sess.run(weight), sess.run(bias)
