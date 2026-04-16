# Representation Learning with Autoencoders (AE & VAE) on Medical MNIST

This repository contains a modular implementation of a Standard Autoencoder (AE) and a Variational Autoencoder (VAE) applied to the Medical MNIST dataset. The project explores unsupervised representation learning, dimensionality reduction, data reconstruction, and sample generation within the medical imaging domain.

**Course:** DSAI 490 - Assignment 1  

---

## 📊 Dataset
This project uses the [Medical MNIST Dataset](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) from Kaggle. It consists of standardized $64 \times 64$ grayscale images across 6 distinct anatomical categories:
* `AbdomenCT`
* `BreastMRI`
* `ChestCT`
* `CXR` (Chest X-Ray)
* `Hand`
* `HeadCT`

---

## 🏗️ Repository Structure
The codebase is designed with strict modularity, separating data pipelines, model architectures, visualization utilities, and unit testing.

```text
medical_mnist_ae_vae/
│
├── models/
│   ├── __init__.py
│   ├── ae.py              # Standard Convolutional Autoencoder
│   └── vae.py             # Variational Autoencoder with custom training step (KL Divergence)
│
├── data/
│   ├── __init__.py
│   └── dataset.py         # Pure tf.data pipelines (resizing, normalization, noise injection)
│
├── utils/
│   ├── __init__.py
│   └── visualization.py   # PCA latent space, reconstruction plotting, loss tracking
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   # Pytest suite for data integrity and model forward-passes
│
└── README.md
```

---

## 🚀 Features

* **Strict `tf.data` Pipeline:** Efficient loading directly from directory structures using TensorFlow's native data API (bypassing memory-heavy `.npz` files). Includes dynamic Gaussian noise injection for denoising tasks.
* **Probabilistic Latent Space:** Custom implementation of the reparameterization trick ($z = \mu + \sigma \odot \epsilon$) for the VAE.
* **Latent Space Visualization:** Automatic Principal Component Analysis (PCA) to map high-dimensional latent vectors ($dim=16$) down to 2D scatter plots for behavioral analysis.
* **Generative & Denoising Capabilities:** Built-in utilities to sample random vectors from $\mathcal{N}(0, I)$ to generate novel anatomical images, and capabilities to map heavily noisy inputs back to clean targets.
* **Automated Unit Testing:** Includes `pytest` scripts to verify tensor shapes, normalization bounds, and architectural integrity.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/medical_mnist_ae_vae.git](https://github.com/YOUR_USERNAME/medical_mnist_ae_vae.git)
   cd medical_mnist_ae_vae
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed. Install the required packages via pip:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn pytest Pillow
   ```

3. **Running the Unit Tests:**
   Verify the codebase is functioning correctly by running the test suite:
   ```bash
   pytest tests/
   ```

---

## 💻 Usage

This codebase is designed to be imported as a module into a training script or a Jupyter/Kaggle Notebook.

**1. Loading Data:**
```python
from data.dataset import get_tf_dataset

# For standard training
dataset = get_tf_dataset('path/to/Hand', batch_size=64, image_size=(64, 64))

# For Denoising Autoencoder
noisy_dataset = get_tf_dataset('path/to/Medical-MNIST', is_noisy=True, noise_factor=0.3)
```

**2. Building and Training the VAE:**
```python
from models.vae import build_vae
from utils.visualization import plot_training_loss, plot_reconstructions, plot_latent_space_2d

vae, encoder, decoder = build_vae(input_shape=(64, 64, 1), latent_dim=16)
vae.compile(optimizer='adam')
history = vae.fit(dataset, epochs=20)

# Visualizations
plot_training_loss(history, is_vae=True)
plot_reconstructions(vae, dataset)
plot_latent_space_2d(encoder, dataset, is_vae=True)
```

---

## 🔬 Key Findings (AE vs VAE)
Extensive experimentation revealed the following architectural differences:
* **Latent Space:** The Standard AE memorizes inputs and plots them arbitrarily, leaving massive "gaps" in the latent space. The VAE successfully utilizes Kullback-Leibler (KL) Divergence to constrain representations into a tight, continuous standard normal distribution ($\mathcal{N}(0, I)$).
* **Generation:** Due to the fragmented space, sampling random vectors from the standard AE yields distorted noise. The VAE's continuous space allows for the generation of highly plausible, novel medical images (e.g., distinct hand phalanges and lung fields) from pure random noise.
* **Denoising:** Both models act as powerful non-linear filters capable of recovering complex anatomical structures from inputs corrupted by heavy Gaussian noise. 