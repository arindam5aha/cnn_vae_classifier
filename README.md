# A CNN VAE Classifier

### Main components
* **Data**: $[x_s, x_f, y]$, where $x_s$ and $x_f$ are the start and final images, and $y$ is the class.
* **Encoder**: $s_f = \phi(s_f|x_f)$, where $s_f$ is the latent state for $x_f$.
* **Decoder**: $x'_f = \Lambda(s_f)$
* **Classifier**: $y' = \psi(s_f)$
* **Predictor**: $s'_f = \tau(s'_f|x_s)$

---

### Implementation

#### Prior
$$\hat{s}_f = \phi_{\theta}(\hat{s}_f|x_f)$$
$$\hat{x}_f = \Lambda_{\theta}(\hat{s}_f)$$
$$\hat{y} = \psi_{\theta}(\hat{s}_f)$$

**Loss Functions:**
$$\mathcal{L}(x_f) = mse(\hat{x}_f, x_f) + \beta_{\phi} D_{KL}(\phi_{\theta}(\hat{s}_f|x_f)||\mathcal{N}(0, I))$$
$$\mathcal{L}(y) = crossentropy(\hat{y}, y)$$
$$\mathcal{L}_{Prior} = \mathcal{L}(y) + \mathcal{L}(x_f)$$

---

#### Prediction
$$s'_f = \tau_{\alpha}(s'_f|x_s)$$
$$y' = \psi_{\alpha}(s'_f)$$

**Loss Functions:**
$$\mathcal{L}(s_f) = mse(s'_{f, \mu}, \hat{s}_{f, \mu}) + \beta_{\tau} D_{KL}(\tau_{\alpha}(s'_f|x_s)||\phi_{\theta}(\hat{s}_f|x_f))$$
$$\mathcal{L}(y) = cross\_entropy(y', y)$$
$$\mathcal{L}_{Prediction} = \mathcal{L}(y) + \mathcal{L}(s_f)$$
