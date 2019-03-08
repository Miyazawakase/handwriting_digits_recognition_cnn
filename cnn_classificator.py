# coding:UTF-8
import numpy as np
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 定义weight和bias，具体看不懂，标准差为啥要0.1，为啥要用截断的正态分布
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 为啥要设置成常量而不是正态分布
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化部分，卷积函数没看懂
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化函数也没看懂
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def predict(image):
    # 这个很重要!!!
    tf.reset_default_graph()
    # 输入的image和label
    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])

    # 不知道为啥，要reshape数据，reshape成4d数据
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一个卷积层，5x5的filter，再max_pool一下，看不懂啊
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积???
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层???
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 添加Dropout，这又是啥，为什么可以防止过拟合
    keep_prob = tf.placeholder("float")
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层，输出一个线性结果??
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

    # 训练
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1)), tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        image_reshape = image.reshape([1, 784])
        saver.restore(sess, "tmp/cnn.ckpt")
        x_ = image_reshape
        res = sess.run(y_conv, feed_dict={x: x_, keep_prob: 1.})
        print('y_prediction: {}\n'.format(res))
        print('Result: {}'.format(np.argmax(res, 1)))
        return np.argmax(res, 1)[0]
