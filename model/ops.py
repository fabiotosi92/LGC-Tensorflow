import tensorflow as tf
import numpy as np


def conv2d(x, kernel_shape, strides=1, relu=True, padding='SAME'):
    W = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable("biases", kernel_shape[3], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
    return x


def conv2d_transpose(x, kernel_shape, strides=1, relu=True):
    W = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable("biases", kernel_shape[2], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    output_shape = [tf.shape(x)[0], tf.shape(x)[1] * strides, tf.shape(x)[2] * strides, kernel_shape[2]]
    with tf.name_scope("deconv"):
        x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
    return x


def encoding_unit(name, inputs, num_outputs):
    with tf.variable_scope('encoding' + str(name)):
        conv = tf.contrib.layers.conv2d(
                    inputs=inputs,
                    num_outputs=num_outputs,
                    kernel_size=3,
                    activation_fn=None
                )
        relu = tf.nn.relu(conv)
        pool = tf.contrib.layers.max_pool2d(relu, 2)

    forward = conv
    return pool, forward


def decoding_unit(number, inputs, num_outputs, forwards=None):
    with tf.variable_scope('decoding' + number):
        conv_transpose = tf.contrib.layers.conv2d_transpose(
                    inputs=inputs,
                    num_outputs=num_outputs*2,
                    kernel_size=3,
                    stride=2,
                    activation_fn=None
                )

        if forwards is not None:
            if isinstance(forwards, (list, tuple)):
                for f in forwards:
                    conv_transpose = tf.concat([conv_transpose, f], axis=3)
            else:
                conv_transpose = tf.concat([conv_transpose, forwards], axis=3)
                
        conv = tf.contrib.layers.conv2d( 
                    inputs=conv_transpose,
                    num_outputs=num_outputs,
                    kernel_size=3,
                    activation_fn=None
                )

        relu = tf.nn.relu(conv)

    return relu
            

def pool_2d(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")


def pad(img):
    hpad = (16 - img.shape[1]%16)%16
    wpad = (16 - img.shape[2]%16)%16
    if hpad+wpad==0:
        return img
    else:
        return np.pad(img, ((0,0),(0,hpad),(0,wpad),(0,0)), 'constant'),hpad,wpad


def depad(img,hpad,wpad):
    return img[:,0:img.shape[1]-hpad,0:img.shape[2]-wpad,:]
