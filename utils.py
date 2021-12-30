import glob
import tensorflow as tf
import numpy as np


def get_image_path(dir):
    file_list = glob.glob(dir + '*.png')
    return file_list


def preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, size=(960, 720),
    #                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image


def init_kernel_left(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[1][0] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_right(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[1][2] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_up(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[0][1] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_down(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[2][1] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_L_up(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[0][0] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_R_up(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[0][2] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_L_down(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[2][0] = -1
    kernel[1][1] = 1
    return kernel


def init_kernel_R_down(shape, dtype=None):
    kernel = np.zeros(shape, dtype=None)
    kernel[2][2] = -1
    kernel[1][1] = 1
    return kernel

