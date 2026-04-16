import tensorflow as tf
from tensorflow.keras import layers, models

def build_ae(input_shape=(64, 64, 1), latent_dim=32):
    # Encoder
    inputs = layers.Input(shape=input_shape, name="ae_input")
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    latent_space = layers.Dense(latent_dim, name="ae_latent")(x)
    encoder = models.Model(inputs, latent_space, name="ae_encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name="ae_output")(x)
    decoder = models.Model(latent_inputs, outputs, name="ae_decoder")

    # Full Autoencoder
    ae_outputs = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, ae_outputs, name="autoencoder")
    
    return autoencoder, encoder, decoder