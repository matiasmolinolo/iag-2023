#!/usr/bin/env python
# coding: utf-8

# Imports


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Lee la imagen y la decodifica. Devuelve un tensor


def load_image(bname, ext="png"):
    fname = bname + "." + ext
    img_file = tf.convert_to_tensor(fname)
    img_str = tf.io.read_file(img_file)
    if ext == "png":
        img_raw = tf.image.decode_png(img_str)
    elif ext == "jpg" or ext == "jpeg":
        img_raw = tf.image.decode_jpeg(img_str)
    else:
        raise Exception('File extension "' + ext + '" not supported')
    return img_raw


# Visualización de la imagen por canales


def plot_image(img):
    channels = np.shape(img)[2]
    _, ax = plt.subplots(nrows=1, ncols=channels + 1, figsize=(20, 5))
    for i in range(0, channels):
        ax[i].imshow(img[:, :, i])
        ax[i].set_title("channel " + str(i), fontsize=12)
    ax_img = ax[channels].imshow(img)
    ax[channels].set_title("image", fontsize=12)
    plt.colorbar(ax_img)
    plt.tight_layout()
    plt.show()


# Visualizar en escala de grises


def plot_grayscale(img):
    plt.imshow(img[:, :, 0], cmap=plt.cm.binary)


def save_grayscale(fname, img):
    plt.imsave(fname, img[:, :, 0], cmap=plt.cm.binary)


# Convierte una imagen rgb a escala de grises (pasa de 3 canales a 1)


def rgb2grayscale(tf_img):
    return tf.image.rgb_to_grayscale(tf_img)


# Convierte una imagen de escala de grises a rgb


def grayscale2rgb(tf_img):
    _convert = tf.image.convert_image_dtype(tf_img, tf.uint8, saturate=True)
    return tf.image.grayscale_to_rgb(_convert)


# Para inicializar una Conv2D con un kernel específico


class KernelInitializer(tf.keras.initializers.Initializer):
    def __init__(self, kern):
        self.kern = kern
        self.shape = (kern.shape[0], kern.shape[1], 1, 1)

    def __call__(self, shape, dtype=None):
        return tf.keras.backend.constant(self.kern.reshape(self.shape))

    def get_config(self):
        return {"kern": self.kern}
