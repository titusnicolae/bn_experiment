from tensorflow.examples.tutorials.mnist import input_data
from time import time
import tensorflow as tf
import numpy as np

def main():
    mnist = input_data.read_data_sets('MNIST_data')

    digit = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None])
    first_layer = tf.nn.relu(tf.add(tf.matmul(digit, tf.Variable(tf.random_normal([784, 200]))), tf.Variable(tf.random_normal([200]))))
    second_layer = tf.add(tf.matmul(first_layer, tf.Variable(tf.random_normal([200, 10]))), tf.Variable(tf.random_normal([10])))
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(second_layer, 1), tf.int32), label), tf.float32))

    softmax_output = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=second_layer)
    loss = tf.reduce_sum(softmax_output)

    optimizer = tf.train.AdamOptimizer()
    optimization_op = optimizer.minimize(loss) 


    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    start = time()
    for i in range(100000):
        digit_batch, label_batch = mnist.train.next_batch(64)
        result = sess.run(optimization_op,
                          feed_dict={digit: digit_batch,
                                     label: label_batch})

    accuracies = [] 
    for i in range(10000): 
        digit_batch, label_batch = mnist.test.next_batch(64)
        accuracy = sess.run(accuracy_op,
                          feed_dict={digit: digit_batch,
                                     label: label_batch})
        #import ipdb; ipdb.set_trace()
        accuracies.append(accuracy) 
    print(np.mean(accuracies))
    
    print("elapsed time", time()-start)
if __name__ == "__main__":
    main()
