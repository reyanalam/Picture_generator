import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, LeakyReLU, Dropout, Reshape
from keras.optimizers import Adam
import os

# Load and preprocess data
def load_train_data():
    try:
        (dataset, _), (_, _) = cifar10.load_data()
        dataset = dataset.astype('float32')
        dataset = (dataset - 127.5) / 127.5  # Normalize to [-1, 1]
        return dataset
    except Exception as e:
        print(f"Error in load_train_data: {e}")
        return None

# Discriminator model
def discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Generator model
def generator(dim):
    model = Sequential()
    nodes = 4 * 4 * 256
    model.add(Dense(nodes, input_dim=dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # Real images labeled as 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Fake images labeled as 0
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)  # Generator wants fake images to be labeled as 1

# Optimizers
discriminator_optimizer = Adam(1e-4)
generator_optimizer = Adam(1e-4)

# Training step
def train_step(real_images, generator, discriminator, batch_size, latent_dim):
    noise = tf.random.normal([batch_size, latent_dim])  # Latent vector
    fake_images = generator(noise, training=True)

    # Train the discriminator
    with tf.GradientTape() as tape_d:
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        d_loss = discriminator_loss(real_output, fake_output)

    discriminator_gradients = tape_d.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # Train the generator
    with tf.GradientTape() as tape_g:
        fake_images = generator(noise, training=True)  # Generate fake images again
        fake_output = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_output)

    generator_gradients = tape_g.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    return d_loss, g_loss

# Main training function
def train(generator, discriminator, dataset, batch_size, latent_dim, epochs):
    for epoch in range(epochs):
        # Create batches of data
        for i in range(0, dataset.shape[0], batch_size):
            real_images = dataset[i:i + batch_size]
            d_loss, g_loss = train_step(real_images, generator, discriminator, batch_size, latent_dim)

        # Print losses
        print(f"Epoch: {epoch+1}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        # Optionally, visualize generated images
        if (epoch + 1) % 10 == 0:
            save_generated_samples(generator, latent_dim, epoch + 1)

def save_generated_samples(generator, latent_dim, epoch, output_dir="generated_images", n_samples=16):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate random latent points
        latent_points = np.random.randn(n_samples, latent_dim)
        if latent_points is None:
            raise ValueError("Latent points generation failed")
        
        # Generate the images from the latent points
        generated_samples = generator.predict(latent_points)
        generated_samples = (generated_samples + 1) / 2.0  # Rescale to [0, 1]

        # Save each image
        for i in range(n_samples):
            img = generated_samples[i, :, :, :]
            img = np.array(img * 255, dtype=np.uint8)  # Convert to [0, 255] range for saving
            
            # Save the image
            img_path = os.path.join(output_dir, f"generated_epoch_{epoch+1}_sample_{i+1}.png")
            plt.imsave(img_path, img)
        
        print(f"Images saved in directory: {output_dir}")
    except Exception as e:
        print(f"Error in save_generated_samples: {e}")

# Initialize models
latent_dim = 100
dis_model = discriminator()
gen_model = generator(latent_dim)

# Load CIFAR-10 data
dataset = load_train_data()
batch_size = 64
epochs = 100

# Train the GAN
#if dataset is not None:
    #train(gen_model, dis_model, dataset, batch_size, latent_dim, epochs)
#else:
    #print("Dataset loading failed, training will not proceed.")

gen_model.save("generator_model.h5")
print("Generator model saved successfully!")
