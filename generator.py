import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained generator
generator = load_model("generator_model.h5")

def generate_images(generator, latent_dim, n_samples=16):
    
    latent_points = np.random.randn(n_samples, latent_dim)
    
    
    generated_images = generator.predict(latent_points)
    
    
    generated_images = (generated_images + 1) / 2.0
    
    # Plot the generated images
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.show()

latent_dim = 100  
generate_images(generator, latent_dim, n_samples=16)
