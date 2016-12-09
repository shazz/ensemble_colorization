#! /usr/bin/python

import tensorflow as tf
import sys 
from matplotlib import pyplot as plt
import skimage
import skimage.io
import numpy as np

filename = sys.argv[1]
phase_train = tf.placeholder(tf.bool, name='phase_train')
uv = tf.placeholder(tf.uint8, name='uv')

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp


def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.mul(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.mul(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model_blue.meta')

    predictions = dict()

    for color in ['red', 'green', 'blue']:
        print 'Restoring session...'
        saver.restore(sess, 'model_%s' % color)
        print 'Session loaded!'

        graph = tf.get_default_graph()
        print 'Loaded default graph'
        
        print 'Processing image %s ...' % filename

        contents = tf.read_file(filename)
        uint8image = tf.image.decode_jpeg(contents, channels=3)
        resized_image = tf.div(tf.image.resize_images(uint8image, (224, 224)), 255)
        img = sess.run(resized_image)

        print'Done processing image!'

        pred = graph.get_tensor_by_name("colornet_1/conv2d_4/Sigmoid:0")

        grayscale = tf.image.rgb_to_grayscale(resized_image)
        grayscale = tf.reshape(grayscale, [1, 224, 224, 1])
        grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
        grayscale_yuv = rgb2yuv(grayscale_rgb)
        grayscale = tf.concat(3, [grayscale, grayscale, grayscale])

        print 'done transforms'
    
        pred_yuv = tf.concat(3, [tf.split(3, 3, grayscale_yuv)[0], pred])
        pred_rgb = yuv2rgb(pred_yuv)

        input_image = sess.run(grayscale)

        feed_dict = {phase_train : False, uv: 3, graph.get_tensor_by_name('concat:0') : input_image}

        print 'Running colornet...'
        pred_, pred_rgb_, colorimage_, grayscale_rgb_ = sess.run(
            [pred, pred_rgb, resized_image, grayscale_rgb], feed_dict=feed_dict)
            
        predictions[color] = pred_rgb_[0]

    # The three arrays:
    # predictions['red'], predictions['green'], predictions['blue']

    # Call recombine code
    # output = recombine(predictions)

    image = concat_images(grayscale_rgb_[0], output)
    image = concat_images(image, img)
    plt.imsave("output.png", image)
