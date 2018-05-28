#!/usr/bin/python3.5
import tensorflow as tf
import numpy as np

def run():
    output = None
    logit_data = np.array([[1., 2., 3., 6.],[2., 4., 5., 6.],[3., 8., 7., 6.]])
    logits = tf.placeholder(tf.float32)
    # TODO: Calculate the softmax of the logits
    # softmax = 
    softmax = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        # output = sess.run(softmax,    )
        output = sess.run(softmax, feed_dict={logits : logit_data})
    return output

