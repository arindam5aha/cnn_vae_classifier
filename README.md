# CNN VAE Classifier

This repository implements a **Convolutional Variational Autoencoder (CVAE)** augmented with a classification branch. The model is designed to learn a compressed latent representation of input images that is simultaneously optimized for both image reconstruction and class label prediction.

## Architecture

The model architecture consists of three primary components:

* **Encoder ($\phi$):** A series of `Conv2d` layers that map input images to a latent distribution (defined by mean $\mu$ and standard deviation $\sigma$).
* **Decoder ($\Lambda$):** A series of `ConvTranspose2d` layers that reconstruct the input image from a sampled latent vector $z$.
* **Classifier ($\psi$):** A Multi-Layer Perceptron (MLP) that maps the latent vector $z$ to a class probability distribution.

## Mathematical Formulation

### Latent Representation
$$\hat{s}_f = \phi_{\theta}(\hat{s}_f|x_f)$$
$$\hat{x}_f = \Lambda_{\theta}(\hat{s}_f)$$
$$\hat{y} = \psi_{\theta}(\hat{s}_f)$$

### Loss Functions
The model is trained using a weighted multi-task objective:

**Total Loss:**
$$\mathcal{L} = \beta \cdot \mathcal{L}_{KL} + \mathcal{L}_{MSE} + \mathcal{L}_{CrossEntropy}$$

* **Reconstruction Loss ($\mathcal{L}_{MSE}$):** $mse(\hat{x}_f, x_f)$
* **Latent Loss ($\mathcal{L}_{KL}$):** $D_{KL}(\phi_{\theta}(\hat{s}_f|x_f) || \mathcal{N}(0, I))$
* **Classification Loss ($\mathcal{L}_{CrossEntropy}$):** $crossentropy(\hat{y}, y)$

## Implementation Details

The implementation utilizes `PyTorch` and `h5py`. 

### Key Features
* **Reparameterization Trick:** Ensures differentiability of the latent space sampling process.
* **Dynamic Weighting:** The $\beta$ hyperparameter allows for controlling the trade-off between latent space regularization and reconstruction fidelity.
* **Gradient Clipping:** Included in the training loop to maintain stability during backpropagation.

## Usage

To train the model, ensure your data is prepared in an HDF5 format and point the `data_file` variable to your dataset. 

```python
# Example setup
cvc = cnn_vae_classifier(input_shape, num_class)
cvc_optimizer = torch.optim.Adam(cvc.parameters(), lr=0.001)

# Training loop
train(cvc, cvc.compute_loss, (train_x, train_y), cvc_optimizer, (val_x, val_y))
