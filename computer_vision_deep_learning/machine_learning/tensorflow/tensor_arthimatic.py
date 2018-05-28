#!/usr/bin/python3.5
# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session
z = tf.subtract(tf.divide(tf.constant(10), tf.constant(2)), tf.cast(tf.constant(1), tf.float64))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
