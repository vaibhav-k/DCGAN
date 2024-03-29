# -*- coding: utf-8 -*-

# import the necessary packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


NUM_EPOCHS = 50
BATCH_SIZE = 128
INIT_LR = 2e-4


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
        alpha: float = 0.2,
    ) -> Sequential:
        """
        Method to build the generator part of the GAN architecture
        """
        # initialize the model along with the input shape
        inputShape = (dim, dim, depth)
        model = Sequential()
        chanDim = -1
        # first set of FC => Leaky ReLU => BN layers
        model.add(Dense(input_dim=input_dimensions, units=output_dimensions))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        # second set of FC => Leaky ReLU => BN layers
        model.add(Dense(dim * dim * depth))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        # reshape the output of the previous layer set
        model.add(Reshape(inputShape))
        # upsample + apply a transposed convolution => Leaky ReLU => BN
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(axis=chanDim))
        # apply another upsample and transposed convolution
        model.add(Conv2DTranspose(channels, (5, 5),
                  strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))
        # return the generator model
        return model

    @staticmethod
    def build_discriminator(
        width: int, height: int, depth: int, alpha: float = 0.2
    ) -> Sequential:
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
        # second set of CONV => RELU layers
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))
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


def define_generator(dim: int, depth: int, channels: int):
    # build the generator
    print("Building the generator...")
    gen = DCGAN.build_generator(dim, depth, channels)
    return gen


def define_discriminator(width: int, height: int, depth: int):
    # build the discriminator
    print("Building the discriminator...")
    disc = DCGAN.build_discriminator(width, height, depth)
    disc_opt = Adam(learning_rate=INIT_LR, beta_1=0.5,
                    decay=INIT_LR / NUM_EPOCHS)

    # compile the discriminator
    disc.compile(loss="binary_crossentropy", optimizer=disc_opt)
    disc.trainable = False
    return disc


def define_dcgan(disc, gen):
    """
    Build the GAN which will be trained in the later part
    """

    # build the adversarial model by combining the generator and discriminator
    print("Building the GAN...")
    gan_input = Input(shape=(100,))
    gan_output = disc(gen(gan_input))
    dcgan = Model(gan_input, gan_output)

    # compile the DCGAN
    gan_opt = Adam(learning_rate=INIT_LR, beta_1=0.5,
                   decay=INIT_LR / NUM_EPOCHS)
    dcgan.compile(loss="binary_crossentropy", optimizer=gan_opt)
    return dcgan


def train_dcgan(disc, gen, gan):
    """
    Train the DCGAN model
    """

    print("Starting training the DCGAN...")

    # randomly generate some benchmark noise
    benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

    # visualize the GAN loss
    gan_loss_list = []

    # loop over the epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting epoch {epoch + 1} of {NUM_EPOCHS}...")

        # compute the number of batches per epoch
        batches_per_epoch = int(train_images.shape[0] / BATCH_SIZE)

        # compute the average of the GAN loss in this epoch
        av_gan_loss = []

        # loop over the batches
        for i in range(batches_per_epoch):
            # initialize an (empty) output path
            p = None

            # select the next batch of images
            image_batch = train_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            # randomly generate noise for the generator to predict on
            noise_for_gen = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            # generate images using the noise + generator model
            genImages = gen.predict(noise_for_gen, verbose=0)

            # concatenate the actual images and the generated images
            X = np.concatenate((image_batch, genImages))

            # construct class labels for the discriminator
            y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
            y = np.reshape(y, (-1,))

            # shuffle the data
            (X, y) = shuffle(X, y)

            # train the discriminator on the data
            discLoss = disc.train_on_batch(X, y)

            # train the generator via the adversarial model
            noise_for_gan = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            fake_labels = [1] * BATCH_SIZE
            fake_labels = np.reshape(fake_labels, (-1,))
            ganLoss = gan.train_on_batch(noise_for_gan, fake_labels)

            # add the loss to the list for visualizing
            av_gan_loss.append(ganLoss)

            # check to see if this is the end of an epoch
            if i == batches_per_epoch - 1:
                # initialize the output path
                p = ["output", "epoch_{}_output.png".format(
                    str(epoch + 1).zfill(4))]
            # check to see if we should visualize the current batch for the epoch
            else:

                # create more visualizations early in the training process
                if epoch < 10 and i % 25 == 0:
                    p = [
                        "output",
                        "epoch_{}_step_{}.png".format(
                            str(epoch + 1).zfill(4), str(i).zfill(5)
                        ),
                    ]

                # visualizations later in the training process are less interesting
                elif epoch >= 10 and i % 100 == 0:
                    p = [
                        "output",
                        "epoch_{}_step_{}.png".format(
                            str(epoch + 1).zfill(4), str(i).zfill(5)
                        ),
                    ]

            # visualize the output of the generator model on our benchmark data
            if p is not None:
                # show loss information
                print(
                    "Step {}_{}: discriminator_loss={:.6f}, "
                    "adversarial_loss={:.6f}".format(
                        epoch + 1, i, discLoss, ganLoss)
                )
                # make predictions on the benchmark noise
                generated_images = gen.predict(benchmark_noise)

                # scale it back to the range [0, 255]
                generated_images = (
                    (generated_images * 127.5) + 127.5).astype("uint8")
                generated_images = np.repeat(generated_images, 3, axis=-1)
                # generate the montage
                vis = build_montages(generated_images, (28, 28), (16, 16))[0]

                # write the visualization to disk
                p = os.path.sep.join(p)
                cv2.imwrite(p, vis)
        gan_loss_list.append(sum(av_gan_loss) / len(av_gan_loss))

    # plot the GAN loss
    plt.plot(gan_loss_list)
    plt.title("The loss of the DCGAN model at every epoch")
    plt.ylabel("DCGAN loss")
    plt.xlabel("Epoch number")
    plt.show()


if __name__ == "__main__":
    # load the images
    train_images = load_data()

    # define the generator and discriminator
    gen = define_generator(7, 64, 1)
    disc = define_discriminator(28, 28, 1)

    # define the DCGAN model
    gan = define_dcgan(disc, gen)

    # train the created model
    train_dcgan(disc, gen, gan)
