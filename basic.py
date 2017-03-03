import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.79

# variable

Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

y = x_data * Weight + bias

loss = tf.reduce_mean(tf.square(y - y_data))
opt = tf.train.GradientDescentOptimizer(0.5)
train = opt.minimize(loss)
init = tf.global_variables_initializer()

### create tensorflow structure start ###

sess = tf.Session()
sess.run(init)


### create tensorflow structure end ###

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(Weight), sess.run(bias)
