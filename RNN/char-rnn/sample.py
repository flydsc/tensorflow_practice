# -*- coding: UTF-8 -*-
import tensorflow as tf
import sys
import numpy as np
import time


def sample(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.log_dir)
        print(ckpt)
        saver.restore(sess, ckpt)

        # initial phrase to warm RNN
        prime = u'你要离开我知道很简单'
        state = sess.run(model.cell.zero_state(1, tf.float32))

        for word in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = data.char2id(word)
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)

        word = prime[-1]
        lyrics = prime
        for i in range(args.gen_num):
            x = np.zeros([1, 1])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(np.argmax(p))
            print word,
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += word
        return lyrics

