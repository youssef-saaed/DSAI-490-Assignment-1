import os
import pytest
import numpy as np
import tensorflow as tf
from PIL import Image

# Import the modules from your codebase
from data.dataset import process_image, add_gaussian_noise, get_tf_dataset
from models.ae import build_ae
from models.vae import build_vae

# ---------------------------------------------------------
# 1. Data Utility Tests
# ---------------------------------------------------------

@pytest.fixture
def dummy_image_file(tmp_path):
    """Fixture to create a temporary dummy JPEG image for testing."""
    img_path = tmp_path / "dummy_test_image.jpeg"
    # Create a random RGB image (dataset process_image converts to grayscale)
    random_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(random_img)
    image.save(img_path)
    return str(img_path)

def test_process_image(dummy_image_file):
    """Test if the image is properly loaded, resized, and normalized."""
    target_size = (64, 64)
    processed = process_image(dummy_image_file, image_size=target_size)
    
    # Check shape (Should be grayscale: 64x64x1)
    assert processed.shape == (*target_size, 1), f"Expected shape {(*target_size, 1)}, got {processed.shape}"
    
    # Check normalization (Values should be between 0.0 and 1.0)
    assert tf.reduce_max(processed) <= 1.0, "Image pixel values exceed 1.0"
    assert tf.reduce_min(processed) >= 0.0, "Image pixel values are below 0.0"

def test_add_gaussian_noise():
    """Test if noise is correctly injected and clipped."""
    # Create a dummy image tensor with pixel values of 0.5
    dummy_img = tf.ones((64, 64, 1), dtype=tf.float32) * 0.5
    noisy_img = add_gaussian_noise(dummy_img, noise_factor=0.2)
    
    # Shape should remain the same
    assert noisy_img.shape == (64, 64, 1), "Noisy image shape mismatch"
    
    # Values should still be clipped between 0.0 and 1.0
    assert tf.reduce_max(noisy_img) <= 1.0, "Noisy image pixel values exceed 1.0"
    assert tf.reduce_min(noisy_img) >= 0.0, "Noisy image pixel values are below 0.0"
    
    # The noisy image should not be identical to the original image
    assert not tf.reduce_all(tf.equal(dummy_img, noisy_img)), "Noise was not added (images are identical)"


# ---------------------------------------------------------
# 2. Autoencoder (AE) Tests
# ---------------------------------------------------------

def test_ae_architecture():
    """Test the instantiation and forward pass of the Standard AE."""
    input_shape = (64, 64, 1)
    latent_dim = 16
    batch_size = 4
    
    ae, encoder, decoder = build_ae(input_shape=input_shape, latent_dim=latent_dim)
    dummy_input = tf.random.uniform((batch_size, *input_shape))
    
    # Test Encoder output shape
    encoded = encoder(dummy_input)
    assert encoded.shape == (batch_size, latent_dim), f"Encoder output shape mismatch. Expected {(batch_size, latent_dim)}"
    
    # Test Decoder output shape
    decoded = decoder(encoded)
    assert decoded.shape == (batch_size, *input_shape), f"Decoder output shape mismatch. Expected {(batch_size, *input_shape)}"
    
    # Test Full AE forward pass
    reconstructed = ae(dummy_input)
    assert reconstructed.shape == (batch_size, *input_shape), "Full AE output shape mismatch"


# ---------------------------------------------------------
# 3. Variational Autoencoder (VAE) Tests
# ---------------------------------------------------------

def test_vae_architecture():
    """Test the instantiation and forward pass of the VAE."""
    input_shape = (64, 64, 1)
    latent_dim = 16
    batch_size = 4
    
    vae, encoder, decoder = build_vae(input_shape=input_shape, latent_dim=latent_dim)
    dummy_input = tf.random.uniform((batch_size, *input_shape))
    
    # Test Encoder output shapes (should return z_mean, z_log_var, z)
    z_mean, z_log_var, z = encoder(dummy_input)
    assert z_mean.shape == (batch_size, latent_dim), "z_mean shape mismatch"
    assert z_log_var.shape == (batch_size, latent_dim), "z_log_var shape mismatch"
    assert z.shape == (batch_size, latent_dim), "z (sampled) shape mismatch"
    
    # Test Decoder output shape
    decoded = decoder(z)
    assert decoded.shape == (batch_size, *input_shape), f"Decoder output shape mismatch. Expected {(batch_size, *input_shape)}"
    
    # Test Full VAE forward pass
    reconstructed = vae(dummy_input)
    assert reconstructed.shape == (batch_size, *input_shape), "Full VAE output shape mismatch"