import tensorflow as tf
from pathlib import Path
import os

def process_image(file_path, image_size=(64, 64)):
    """Reads and normalizes an image."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1) # Grayscale for Medical MNIST
    img = tf.image.resize(img, image_size)
    img = img / 255.0 # Normalize to [0, 1]
    return img

def add_gaussian_noise(img, noise_factor=0.2):
    """Injects noise for denoising applications."""
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    noisy_img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return noisy_img

def get_tf_dataset(data_dir, batch_size=32, image_size=(64, 64), is_noisy=False, noise_factor=0.2):
    """
    Creates a tf.data.Dataset. 
    data_dir can be a specific anatomical folder (e.g., '.../Hand') or the root for combined data.
    """
    # Find all jpegs in directory (and subdirectories if root is provided)
    file_pattern = os.path.join(data_dir, '**', '*.jpeg')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    def process_path(file_path):
        img = process_image(file_path, image_size)
        if is_noisy:
            noisy_img = add_gaussian_noise(img, noise_factor)
            return noisy_img, img # Input is noisy, Target is clean
        return img, img # Input is clean, Target is clean

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset