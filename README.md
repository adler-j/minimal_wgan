Minimal Wasserstein GAN
=======================

This is a minimal implementation of Wasserstein GAN in Tensorflow applied to MNIST.

Some example generated numbers:

![WGAN results](https://user-images.githubusercontent.com/2202312/32316459-9963dc90-bfb0-11e7-9bbd-3d60ed1ad287.png)

How to run
----------

Simply clone the directory and run the file [`wgan_mnist.py`](wgan_mnist.py). Results will be displayed in real time, while full training takes about an hour.

Implementation details
----------------------

The implementation follows [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028), using the network from [the accompanying code](https://github.com/igul222/improved_wgan_training). In particular both the generator and discriminator uses 3 convolutional layers with 5x5 convolutions.
