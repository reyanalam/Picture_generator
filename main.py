import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D 
from keras.layers import Flatten
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU 
from keras.layers import Dropout
from keras.layers import Reshape
from keras.optimizers import Adam

from keras.datasets.cifar10 import load_data

def load_train_data():
    (dataset,_),(_,_) = load_data()
    dataset = dataset.astype('float32')
    dataset = (dataset-127.5)/127.5
    return dataset

def discriminator(in_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(64,(3,3),padding='same',input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.002, beta_1=0.5),metrics=['accuracy'])
    
    return model

def generator(dim):
    model = Sequential()
    nodes=4*4*256
    model.add(Dense(nodes,input_dim=dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4,4,256)))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3,(3,3),activation='tanh',padding='same'))
    return model

def gan(gen_model,dis_model):
    dis_model.trainable = False
    model = Sequential()
    model.add(gen_model)
    model.add(dis_model)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.002, beta_1=0.5))
    return model

def load_real_data(dataset,sample_size):
    j = np.random.randint(0,dataset.shape[0],sample_size)
    x = dataset[j]
    y = np.ones((sample_size,1))
    return x,y

def latent_point(dim,sample_size):
    x_input = np.random.randn(dim*sample_size)
    x = x_input.reshape(sample_size,dim)
    return x

def load_fake_data(gen_model,dim,sample_size):
    x_input = latent_point(dim,sample_size)
    x = gen_model.predict(x_input)
    y = np.zeros((sample_size,1))
    return x,y

def train(g_model,d_model,gan_model,dataset,dim,epoch=200,batch=128):
    batch_perepoch = int(dataset.shape[0]/batch)
    half_batch = int(batch/2)

    for i in range(epoch):
    
        for j in range(batch_perepoch):
            x_real,y_real = load_real_data(dataset,half_batch)
            d1_loss,_ = d_model.train_on_batch(x_real,y_real)
            
            x_fake,y_fake = load_fake_data(g_model,dim,half_batch)
            d2_loss,_ = d_model.train_on_batch(x_fake,y_fake)
            
            x_gan = latent_point(dim,batch)
            y_gan = np.ones((batch,1))
            gan_loss,_ = gan_model.train_on_batch(x_gan,y_gan)
            
            print(f'>{i+1}, {j+1}, {batch_perepoch}, d1={d1_loss:.3f}, d2={d2_loss:.3f}, g={gan_loss:.3f}')


        if (i + 1) % 10 == 0:
            visualize_generated_samples(g_model, dim, i + 1)

def visualize_generated_samples(generator, latent_dim, epoch, n_samples=16):
    # Generate latent points
    latent_points = latent_point(latent_dim, n_samples)
    # Generate fake samples
    generated_samples = generator.predict(latent_points)
    # Rescale pixel values to the range [0, 1] if needed
    generated_samples = (generated_samples + 1) / 2.0

    # Plot the generated samples
    plt.figure(figsize=(8, 8))
    for i in range(n_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i, :, :, :])  # Adjust if dataset is RGB
        plt.axis('off')
    plt.suptitle(f'Generated Samples at Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.show()

dim = 100
dis_model = discriminator()
gen_model = generator(dim)
gan_model = gan(gen_model,dis_model)
dataset = load_train_data()

train(gen_model,dis_model,gan_model,dataset,dim)






