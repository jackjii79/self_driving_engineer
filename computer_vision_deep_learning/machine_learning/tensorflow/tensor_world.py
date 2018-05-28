#!/usr/bin/python3.5
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello_constant = tf.constant('Hello World')
with tf.Session() as sess:
 output = sess.run(hello_constant)
 print(output)
