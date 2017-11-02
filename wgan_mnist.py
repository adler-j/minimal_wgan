import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

session = tf.InteractiveSession()


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


def generator(z):
    with tf.variable_scope('generator'):
        z = layers.fully_connected(z, num_outputs=4096)
        z = tf.reshape(z, [-1, 4, 4, 256])

        z = layers.conv2d_transpose(z, num_outputs=128, kernel_size=5, stride=2)
        z = z[:, :7, :7, :]
        z = layers.conv2d_transpose(z, num_outputs=64, kernel_size=5, stride=2)
        z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=2,
                                    activation_fn=tf.nn.sigmoid)

        return z


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = layers.conv2d(x, num_outputs=64, kernel_size=5, stride=2,
                          activation_fn=leaky_relu)
        x = layers.conv2d(x, num_outputs=128, kernel_size=5, stride=2,
                          activation_fn=leaky_relu)
        x = layers.conv2d(x, num_outputs=256, kernel_size=5, stride=2,
                          activation_fn=leaky_relu)

        x = layers.flatten(x)
        x = layers.fully_connected(x, num_outputs=1, activation_fn=None)
        return x


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, 28, 28, 1])
    z = tf.placeholder(tf.float32, [None, 128])


x_generated = generator(z)

d_true = discriminator(x_true, reuse=False)
d_generated = discriminator(x_generated, reuse=True)

with tf.name_scope('loss'):
    g_loss = tf.reduce_mean(d_generated)
    d_loss = tf.reduce_mean(d_true) - tf.reduce_mean(d_generated)

    epsilon = tf.random_uniform([], 0.0, 1.0)
    x_hat = epsilon * x_true + (1 - epsilon) * x_generated
    d_hat = discriminator(x_hat, reuse=True)

    ddx = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0))

    d_loss = d_loss + 10 * ddx

with tf.name_scope('optimizer'):
    generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    train_generator = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.9).minimize(g_loss, var_list=generator_vars)
    discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    train_discriminator = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.9).minimize(d_loss, var_list=discriminator_vars)

tf.global_variables_initializer().run()

plt.figure('results')
z_validate = np.random.randn(1, 128)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    images = batch[0].reshape([-1, 28, 28, 1])
    z_train = np.random.randn(50, 128)

    session.run(train_generator, feed_dict={z: z_train})
    for j in range(5):
        session.run(train_discriminator, feed_dict={x_true: images, z: z_train})

    if i % 100 == 0:
        print('iter={}/20000'.format(i))
        generated = x_generated.eval(feed_dict={z: z_validate}).squeeze()
        plt.imshow(generated, clim=[0, 1])
        plt.pause(0.001)
