import numpy as np
import tensorflow as tf
import helpers
import load

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6


def next_feed(sequences):
    EOS = 1
    encoder_inputs_, _ = helpers.batch(sequences)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in sequences]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in sequences]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

tf.reset_default_graph()
with tf.Session() as sess:

    PAD = 0
    EOS = 1

    vocab_size = 2498
    input_embedding_size = 20

    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units

    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
    )

    del encoder_outputs

    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,

        initial_state=encoder_final_state,

        dtype=tf.float32, time_major=True, scope="plain_decoder",
    )

    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

    decoder_prediction = tf.argmax(decoder_logits, 2)

    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )

    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)


    sess.run(tf.global_variables_initializer())

    batch_size = 100

    # batches = helpers.random_sequences(length_from=3, length_to=8,
    #                                    vocab_lower=2, vocab_upper=10,
    #                                    batch_size=batch_size)
    #
    # print len(next(batches))

    loss_track = []
    max_batches = 3001
    batches_in_epoch = 1000
    b_size = 10
    max_epoch = 1000000

    all_data = load.seq()


    for i in range(len(all_data)):
        try:
            fd = next_feed(all_data[i: min(len(all_data), i + 10)])
            for epoch in range(max_epoch):# print('minibatch loss: {}'.format(sess.run(loss, feed_dict={encoder_inputs: encoder_inputs_[ep_idx*10: (ep_idx+1)*10-1], decoder_inputs: decoder_inputs_[ep_idx*10: (ep_idx+1)*10-1], decoder_targets: decoder_targets_[ep_idx*10: (ep_idx+1)*10-1]})))
                _, l = sess.run([train_op, loss], fd)
                if epoch % 1000 == 0:
                    print('loss: {}'.format(sess.run(loss, fd)))
                    testfd = next_feed(load.test())
                    predict_ = sess.run(decoder_prediction, testfd)
                    print load.getch(predict_.T)
        except KeyboardInterrupt:
            print('training interrupted')
        i += 10

