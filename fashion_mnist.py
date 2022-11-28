# -*- coding: utf-8 -*-

# import the necessary packages
import os
import cv2
import numpy as np
from imutils import build_montages
from numpy.random import uniform
from sklearn.utils import shuffle
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential


class DCGAN:
    """
    Class to instantiate a Deep Convolutional Generative Adversarial Networks.
    This architecture was introduced in the paper https://arxiv.org/abs/1511.06434.
    """

    @staticmethod
    def build_generator(
        dim: int,
        depth: int,
        channels: int = 1,
        input_dimensions: int = 100,
        output_dimensions: int = 512,
    ):
        """
        Method to build the generator part of the GAN architecture
        """
        # initialize the model along with the input shape
        inputShape = (dim, dim, depth)
        model = Sequential()
        chanDim = -1
        # first set of FC => RELU => BN layers
        model.add(Dense(input_dim=input_dimensions, units=output_dimensions))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # second set of FC => RELU => BN layers
        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # reshape the output of the previous layer set
        model.add(Reshape(inputShape))
        # upsample + apply a transposed convolution, RELU, and BN
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # apply another upsample and transposed convolution
        model.add(Conv2DTranspose(channels, (5, 5),
                  strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))
        # return the generator model
        return model

    @staticmethod
    def build_discriminator(width: int, height: int, depth: int, alpha: int = 0.2):
        """
        Method to build the discriminator part of the GAN architecture
        """
        # initialize the model along with the input shape
        inputShape = (height, width, depth)
        model = Sequential()
        # first set of CONV => RELU layers
        model.add(
            Conv2D(32, (5, 5), padding="same", strides=(
                2, 2), input_shape=inputShape)
        )
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        # second set of CONV => RELU layers
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        # sigmoid activation to output a single value
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        # return the discriminator model
        return model


def load_data() -> np.ndarray:
    """
    Load the Fashion MNIST dataset and do preliminary data manipulation
    """
    # load the Fashion MNIST dataset
    print("Loading the Fashion MNIST dataset...")
    ((X_train, _), (X_test, _)) = fashion_mnist.load_data()
    images = np.concatenate([X_train, X_test])
    # stack the training and testing data points so we have additional
    # training data
    images = np.expand_dims(images, axis=-1)
    # scale the images into the range [-1, 1]
    images = (images.astype("float") - 127.5) / 127.5
    return images


def build_generator_discriminator():
    """
    Build the Generator and Discriminator models for use in the downstream GAN
    """
    # build the generator
    print("Building the generator...")
    gen = DCGAN.build_generator(7, 64, channels=1)

    # build the discriminator
    print("Building the discriminator...")
    disc = DCGAN.build_discriminator(28, 28, 1)

    # compile the model
    disc.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS),
    )

    # setting the discriminator to not be trainable
    disc.trainable = False
    return gen, disc


def build_gan(gen, disc):
    """
    Build the GAN which will be trained in the later part
    """
    # build the adversarial model
    print("Building the GAN...")
    ganInput = Input(shape=(100,))
    ganOutput = disc(gen(ganInput))

    # combine the generator and discriminator together
    gan = Model(ganInput, ganOutput)

    # compile the GAN
    gan.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS),
    )
    return gan


def make_noise(low: int = -1, high: int = 1, size: tuple = None):
    """
    Generate random noise vectors for the generative model training
    """
    return uniform(low, high, size)


def write_visualization(p, vis):
    """
    Write the visualization to disk
    """
    p = os.path.sep.join(p)
    cv2.imwrite(p, vis)


def train_gan(images, gen, disc, gan):
    """
    Train the GAN model defined
    """
    print("Starting training...")
    # randomly generate some benchmark noise (to visualize how the generative modeling is learning)
    benchmark_noise = make_noise(-1, 1, (256, 100))

    # loop over the epochs
    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1} of {NUM_EPOCHS}...")
        # compute the number of batches per epoch
        batches_per_epoch = int(images.shape[0] / BATCH_SIZE)

        # loop over the batches
        for i in range(batches_per_epoch):
            # initialize the output path
            p = None

            # select the next batch of images
            batch_of_images = images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            # randomly generate noise for the generator to predict on
            generated_noise = make_noise(-1, 1, (BATCH_SIZE, 100))

            # generate images using the noise
            generated_images = gen.predict(generated_noise, verbose=0)

            # concatenate the actual images and the generated images
            X = np.concatenate((batch_of_images, generated_images))
            # construct class labels for the discriminator
            y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
            y = np.reshape(y, (-1,))

            # shuffle the data
            (X, y) = shuffle(X, y)

            # train the discriminator on the data
            discriminator_loss = disc.train_on_batch(X, y)

            # train the generator via the adversarial model
            # generate random noise
            generated_noise = make_noise(-1, 1, (BATCH_SIZE, 100))
            fake_labels = [1] * BATCH_SIZE
            fake_labels = np.reshape(fake_labels, (-1,))
            # train the generator
            gan_loss = gan.train_on_batch(generated_noise, fake_labels)

            # check to see if this is the end of an epoch
            if i == batches_per_epoch - 1:
                # initialize the output path
                p = [
                    "./", "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]

            # check to see if we should visualize the current batch for the epoch
            else:
                # create more visualizations early in the training process
                if epoch < 10 and i % 25 == 0:
                    p = [
                        "./",
                        "epoch_{}_step_{}.png".format(
                            str(epoch + 1).zfill(4), str(i).zfill(5)
                        ),
                    ]

                # visualizations later in the training process are less interesting
                elif epoch >= 10 and i % 100 == 0:
                    p = [
                        "./",
                        "epoch_{}_step_{}.png".format(
                            str(epoch + 1).zfill(4), str(i).zfill(5)
                        ),
                    ]

            # check to see if we should visualize the output of the generator model on our benchmark data
            if p is not None:
                # show loss information
                print(
                    "Step {}_{}: discriminator_loss={:.6f}, "
                    "adversarial_loss={:.6f}".format(
                        epoch + 1, i, discriminator_loss, gan_loss
                    )
                )

                # make predictions on the benchmark noise
                images = gen.predict(benchmark_noise)
                # scale it back to the range [0, 255]
                images = ((images * 127.5) + 127.5).astype("uint8")
                # generate the montage
                images = np.repeat(images, 3, axis=-1)
                vis = build_montages(images, (28, 28), (16, 16))[0]

                write_visualization(p, vis)


if __name__ == "__main__":
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    INIT_LR = 2e-4

    # load the Fashion MNIST dataset
    images = load_data()

    # build the generator and discriminator models
    gen, disc = build_generator_discriminator()

    # build the adversarial model
    gan = build_gan(gen, disc)

    train_gan(images, gen, disc, gan)
