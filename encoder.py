import tensorflow as tf
import numpy as np


# 创建encoder/decoder
class endecoder(object):
    def __init__(self):
        self.trainable = True
        n_input = 1
        n_hidden_1 = 50
        n_hidden_2 = 100
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), trainable=self.trainable),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), trainable=self.trainable),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), trainable=self.trainable),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]), trainable=self.trainable),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), trainable=self.trainable),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]), trainable=self.trainable),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), trainable=self.trainable),
            'decoder_b2': tf.Variable(tf.random_normal([n_input]), trainable=self.trainable),
        }

    # 构建编码器
    def encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    # 构建解码器
    def decoder(self, encoder_op):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_op, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2

    def autoencoder(self, x):
        encoder_op = self.encoder(x)
        decoder_op = self.decoder(encoder_op)
        loss = tf.losses.mean_squared_error(x, decoder_op)
        return encoder_op, loss
