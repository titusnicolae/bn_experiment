from tensorflow.examples.tutorials.mnist import input_data
from time import time
import tensorflow as tf
import numpy as np


def selu(x, name="selu"):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def diy(digit, activation=tf.nn.relu):
    first_layer = activation(tf.add(tf.matmul(digit, tf.Variable(tf.random_normal(
        [784, 200], mean=0, stddev=1.0 / 784))), tf.Variable(tf.zeros([200]))))
    second_layer = tf.add(tf.matmul(first_layer, tf.Variable(tf.random_normal(
        [200, 10], mean=9, stddev=1.0 / 20))), tf.Variable(tf.zeros([10])))
    return second_layer


def fully_connected(digit):
    batch_norm = tf.contrib.layers.batch_norm  # 0.973 0.958
    first_layer = tf.contrib.layers.fully_connected(
        digit,       200, normalizer_fn=batch_norm)
#    second_layer = tf.contrib.layers.fully_connected(first_layer,  50, normalizer_fn=batch_norm)
    third_layer = tf.contrib.layers.fully_connected(
        first_layer,  10, normalizer_fn=batch_norm)
    return third_layer


def bn_network(digit, is_training):
    first_layer = tf.contrib.layers.batch_norm(tf.add(tf.matmul(digit, tf.Variable(tf.random_normal(
        [784, 200], mean=0, stddev=1.0 / 784))), tf.Variable(tf.zeros([200]))), center=True, scale=True, is_training=is_training)

    second_layer = tf.contrib.layers.batch_norm(tf.add(tf.matmul(first_layer, tf.Variable(tf.random_normal(
        [200, 10], mean=9, stddev=1.0 / 20))), tf.Variable(tf.zeros([10]))), center=True, scale=True, is_training=is_training)

    return second_layer

def main():
    mnist = input_data.read_data_sets('MNIST_data')

    digit = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)

    last_layer = bn_network(digit, is_training)
    accuracy_op = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(tf.argmax(last_layer, 1), tf.int32), label), tf.float32))

    softmax_output = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=last_layer)
    loss = tf.reduce_sum(softmax_output)

    optimizer = tf.train.AdamOptimizer()
    optimization_op = optimizer.minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    start = time()
    # log_file = open("output.txt"
    for i in range(10000):
        digit_batch, label_batch = mnist.train.next_batch(64)
        _ = sess.run(optimization_op,
                     feed_dict={digit: digit_batch,
                                label: label_batch,
                                is_training: True})

        """jj
        digit_batch, label_batch = mnist.test.next_batch(64)
        accuracy = sess.run(accuracy_op,
                          feed_dict={digit: digit_batch,
                                     label: label_batch})
        print(accuracy)
        """


    for test_bs in [1,2,4,8,16,32,64,128,256,512,1024,2048,4196]:
#    for test_bs in [64]:
        accuracies = []
        for i in range(int(10000 / test_bs)):
            digit_batch, label_batch = mnist.test.next_batch(test_bs)
            accuracy = sess.run(accuracy_op,
                                feed_dict={digit: digit_batch,
                                           label: label_batch,
                                           is_training: False})
            accuracies.append(accuracy)
        print(test_bs, np.mean(accuracies))
    print("elapsed time", time() - start)


if __name__ == "__main__":
    main()
