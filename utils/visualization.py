import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_reconstructions(model, dataset, n=5, title="Reconstructions"):
    """Compares original vs reconstructed outputs."""
    images, _ = next(iter(dataset.take(1)))
    reconstructions = model.predict(images[:n])
    
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
    generated_images = decoder.predict(random_latent_vectors)
    
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def plot_latent_space_2d(encoder, dataset, is_vae=False, title="Latent Space 2D"):
    """Visualizes 2D Latent space (Requires model trained with latent_dim=2)."""
    images, _ = next(iter(dataset.unbatch().batch(500).take(1)))
    
    if is_vae:
        z_mean, _, _ = encoder.predict(images)
        latent = z_mean
    else:
        latent = encoder.predict(images)
        
    plt.figure(figsize=(6, 6))
    plt.scatter(latent[:, 0], latent[:, 1], alpha=0.7)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title)
    plt.show()

def plot_denoising_results(model, noisy_dataset, n=5):
    """Plots Noisy Input -> Clean Reconstruction."""
    noisy_images, clean_images = next(iter(noisy_dataset.take(1)))
    reconstructions = model.predict(noisy_images[:n])
    
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