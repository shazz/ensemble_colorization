#! /usr/bin/python

import tensorflow as tf
import sys 
from matplotlib import pyplot as plt
import skimage
import skimage.io

filename = sys.argv[1]
phase_train = tf.placeholder(tf.bool, name='phase_train')
uv = tf.placeholder(tf.uint8, name='uv')
grayscale = tf.placeholder(tf.float32, [1,224,224,3], name='grayscale')

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
  
    print 'Restoring session...'
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print 'Session loaded!'

    graph = tf.get_default_graph()
    print 'Loaded default graph'

    print 'Pxrocessing image %s ...' % filename

    contents = tf.read_file(filename)
    uint8image = tf.image.decode_jpeg(contents, channels=3)

    resized_image = tf.image.resize_images(uint8image, (224, 224))
    sess.run(uint8image)

    print'Done processing image!'

    init = tf.initialize_all_variables()
    init_local = tf.initialize_all_variables()
    sess.run(init)
    sess.run(init_local)
    print "variables initialized"

    pred = graph.get_tensor_by_name("colornet_1/conv2d_4/Sigmoid:0")
    print pred

    gray = tf.image.rgb_to_grayscale(resized_image)
    gray = tf.reshape(gray, [1, 224, 224, 1])
    gray_rgb = tf.image.grayscale_to_rgb(gray)
    gray_yuv = rgb2yuv(gray_rgb)
    gray = tf.concat(3, [gray, gray, gray])

    print 'done transforms'
    
    pred_yuv = tf.concat(3, [tf.split(3, 3, gray_yuv)[0], pred])
    pred_rgb = yuv2rgb(pred_yuv)

    input_image = sess.run(gray)

    feed_dict = {phase_train : False, uv: 3, graph.get_tensor_by_name('concat:0') : input_image}

    print 'Running colornet...'
    pred_, pred_rgb_, colorimage_, gray_rgb_ = sess.run(
        [pred, pred_rgb, resized_image, gray_rgb], feed_dict=feed_dict)
    pred_yuv = tf.concat(3, [tf.split(3, 3, gray_yuv)[0], pred])
    pred_rgb = yuv2rgb(pred_yuv)
    
    image = concat_images(resized_image, pred)
    plt.imsave("output.png", image)
