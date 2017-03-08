import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


def add_layer(input_data, in_dim, out_dim, act_fun=None, ):
    Weight = tf.Variable(tf.random_normal([in_dim, out_dim]))
    bias = tf.Variable(tf.zeros([1, out_dim]) + 0.1, )
    Wx_plus_b = tf.matmul(input_data, Weight) + bias
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if act_fun:
        output = act_fun(Wx_plus_b)
    else:
        output = Wx_plus_b
    return output


digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs, 64, 50, act_fun=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, act_fun=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
# sess.run(init)
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i % 50 == 0:
            print(sess.run(cross_entropy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1}))
