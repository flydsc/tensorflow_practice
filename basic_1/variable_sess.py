import tensorflow as tf

stat = tf.Variable(0, name='counter')

one = tf.constant(1)

new = tf.add(stat, one)

update = tf.assign(stat, new)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for _ in range(200):
        sess.run(update)
        print sess.run(stat)

