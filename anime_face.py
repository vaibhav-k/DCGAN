import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from skimage.transform import resize
from multiprocessing import Pool, cpu_count


def load_image(filename):
    """
    Load an image and resize it to 64x64
    """
    image = imageio.v3.imread(os.path.join(dir_path, filename))
    image_resized = resize(image, (64, 64), preserve_range=True)
    return image_resized.astype(np.uint8)


def image_generator(batch_size=32):
    """
    Yield batches of images
    """
    # Get the list of JPEG files in the directory
    filenames = [filename for filename in os.listdir(
        dir_path) if filename.endswith(".jpg") or filename.endswith(".jpeg")]
    # Define the pool of worker processes for multiprocessing
    pool = Pool(processes=cpu_count())
    # Iterate over the filenames in batches of batch_size
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i+batch_size]
        # Use the pool to load and resize the images in parallel
        batch_images = pool.map(load_image, batch_filenames)
        # Convert the batch of images to a NumPy array and yield it
        yield np.array(batch_images)


def convert_images_npy(dir_path):
    batch_size = 32
    # iterate over the batches of images
    generator = image_generator(batch_size=batch_size)
    i = 0
    while i <= len(os.listdir(dir_path)):
        images = next(generator)
        i += batch_size
        print(f"{i} out of {len(os.listdir(dir_path))}")
        np.save(f"./animefaces_data/batch_{i}.npy", images)


def merge_npys(dir_path):
    npy_files = [os.path.join(dir_path, f)
                 for f in os.listdir(dir_path) if f.endswith(".npy")]
    data = np.load(npy_files[0], allow_pickle=True)
    for f in npy_files[1:]:
        data = np.concatenate((data, np.load(f, allow_pickle=True)), axis=0)
    np.save("./keras_models/merged_data.npy", data)


def load_data(npy_path):
    return np.load(npy_path)


def normalize_save_merged_array(X):
    """
    Normalize pixel values between -1 and 1 and then save the array
    """
    np.save("./keras_models/merged_data.npy",
            (X.astype("float32") * 127.5 + 127.5).astype(np.uint8))


def build_generator():
    """
    Define the generator
    """
    generator = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, input_dim=100),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(64 * 64 * 3, activation="tanh"),
            tf.keras.layers.Reshape((64, 64, 3)),
        ]
    )

    return generator


def build_and_compile_disriminator():
    """
    Define and compile the discriminator
    """

    # build the discriminator
    discriminator = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', input_shape=(64, 64, 3)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        # Second convolutional layer
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        # Third convolutional layer
        tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        # Fourth convolutional layer
        tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        # Flatten the output of the final convolutional layer
        tf.keras.layers.Flatten(),
        # Final output layer, outputting a single scalar value
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the discriminator
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
    )

    return discriminator


def build_and_compile_GAN(discriminator, generator):
    """
    Define the GAN
    """
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return gan


def generate_noise(n_samples, noise_dim):
    """
    Generate noise for the generator
    """
    return np.random.normal(0, 1, size=(n_samples, noise_dim))


def train_GAN(discriminator, generator, gan, X):
    """
    Training loop
    """
    epochs = 10000
    batch_size = 32
    sample_interval = 1000

    # create a folder to save the generated images
    os.makedirs("./anime_face_generated_images", exist_ok=True)

    # train the GAN
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_images = X[idx]
        noise = generate_noise(batch_size, 100)
        fake_images = generator.predict(noise)
        discriminator_loss_real = discriminator.train_on_batch(
            real_images, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(
            fake_images, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * \
            np.add(discriminator_loss_real, discriminator_loss_fake)

        # train generator
        noise = generate_noise(batch_size, 100)
        generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # save generated images
        if epoch % sample_interval == 0:
            print(
                f"Epoch {epoch}: discriminator_loss = {discriminator_loss},\
                generator_loss = {generator_loss}"
            )
            noise = generate_noise(25, 100)
            generated_images = generator.predict(noise)
            generated_images = 0.5 * generated_images + 0.5
            fig, axs = plt.subplots(5, 5)
            cnt = 0
            for i in range(5):
                for j in range(5):
                    axs[i, j].imshow(generated_images[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(
                "./anime_face_generated_images/anime_faces_%d.png" % epoch)
            plt.close()

    # return the trained models
    return discriminator, generator, gan


def save_models(discriminator, generator, gan):
    """
    Save the trained models to the disk
    """
    # make the directory to store models if one does not exist
    os.makedirs("./keras_models/", exist_ok=True)
    discriminator.save("./keras_models/discriminator_model.h5")
    generator.save("./keras_models/generator_model.h5")
    gan.save("./keras_models/gan_model.h5")


def load_models(discriminator, generator, gan):
    """
    Load the Keras models saved to the disk
    """
    from tensorflow.keras.models import load_model

    discriminator = load_model(
        "./keras_models/discriminator_model.h5")
    generator = load_model("./keras_models/generator_model.h5")
    gan = load_model("./keras_models/gan_model.h5")
    return discriminator, generator, gan


if __name__ == "__main__":
    X = load_data("./keras_models/merged_data.npy")
    # normalize_save_merged_array(X)
    generator = build_generator()
    discriminator = build_and_compile_disriminator()
    gan = build_and_compile_GAN(discriminator, generator)
    discriminator, generator, gan = train_GAN(discriminator, generator, gan, X)
    save_models(discriminator, generator, gan)
