# -*- coding: UTF-8 -*-
import tensorflow as tf
import os


def train(data, model, configs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        max_iter = configs.n_epoch * (data.total_len // configs.seq_length) // configs.batch_size
        for i in range(max_iter):
            learning_rate = configs.learning_rate * (configs.decay_rate ** (i // configs.decay_steps))
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch, model.target_data: y_batch, model.lr: learning_rate}
            train_loss, summary, _ = sess.run([model.cost, model.last_state, model.train_op], feed_dict)
            if i % 10 == 0:
                print('Step:{}/{}, training_loss:{:4f}'.format(i, max_iter, train_loss))
            if i % 2000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(
                    configs.log_dir, 'lyrics_model.ckpt'), global_step=i)

