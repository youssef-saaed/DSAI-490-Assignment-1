import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

def plot_reconstructions(model, dataset, n=5, title="Reconstructions"):
    """Compares original vs reconstructed outputs."""
    images, _ = next(iter(dataset.take(1)))
    reconstructions = model.predict(images[:n], verbose=0)
    
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")
        
        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].squeeze(), cmap="gray")
        plt.title("Recon")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def plot_generated_samples(decoder, latent_dim, n=5, title="Generated Samples"):
    """Generates new samples from the VAE latent space."""
    random_latent_vectors = tf.random.normal(shape=(n, latent_dim))
    generated_images = decoder.predict(random_latent_vectors, verbose=0)
    
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def plot_latent_space_2d(encoder, dataset, is_vae=False, title="Latent Space 2D"):
    """
    Visualizes the latent space. If the latent dimension is > 2, 
    it applies PCA to reduce it to 2 dimensions for plotting.
    """
    # Unbatch and take a larger sample size (e.g., 1000 images) for a denser scatter plot
    images, _ = next(iter(dataset.unbatch().batch(1000).take(1)))
    
    if is_vae:
        z_mean, _, _ = encoder.predict(images, verbose=0)
        latent_representations = z_mean
    else:
        latent_representations = encoder.predict(images, verbose=0)
        
    latent_dim = latent_representations.shape[1]
    
    # Apply PCA if dimensionality is higher than 2
    if latent_dim > 2:
        print(f"Latent dimension is {latent_dim}. Applying PCA to reduce to 2D...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_representations)
        xlabel, ylabel = "Principal Component 1", "Principal Component 2"
    elif latent_dim == 2:
        latent_2d = latent_representations
        xlabel, ylabel = "z[0]", "z[1]"
    else:
        raise ValueError("Latent dimension must be at least 2.")
        
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_denoising_results(model, noisy_dataset, n=5):
    """Plots Noisy Input -> Clean Reconstruction."""
    noisy_images, clean_images = next(iter(noisy_dataset.take(1)))
    reconstructions = model.predict(noisy_images[:n], verbose=0)
    
    plt.figure(figsize=(15, 4))
    for i in range(n):
        # Original Clean
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(clean_images[i].numpy().squeeze(), cmap="gray")
        plt.title("Clean Ground Truth")
        plt.axis("off")
        
        # Noisy Input
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy_images[i].numpy().squeeze(), cmap="gray")
        plt.title("Noisy Input")
        plt.axis("off")
        
        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(reconstructions[i].squeeze(), cmap="gray")
        plt.title("Denoised")
        plt.axis("off")
    plt.suptitle("Denoising Performance")
    plt.show()