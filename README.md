# A CNN VAE Classifier

<b>Main components:</b> <br>
Data : $[x_s, x_f, y]$, where, $x_s$ and $x_f$ are the start and final images, $y$ is the class<br>
Encoder : $s_{f} = \phi(s_{f}|x_{f})$, where, $s_f$ is the latent state for $x_f$<br>
Decoder : $x^{'}_{f} = \Lambda(s_f)$<br>
Classifier : $y^{'} = \psi(s_{f})$<br>
Predictor  : $s'_{f} = \tau(s'_f|x_s)$<br>

<b>Implimentation:</b><br>
<b>Prior:</b><br>
$\hat{s}_{f} = \phi_{\theta}(\hat{s}_f|x_f)$<br>
$\hat{x}_{f} = \Lambda_{\theta}(\hat{s}_f)$<br>
$\hat{y} = \psi_{\theta}(\hat{s}_f)$<br>

Loss Functions:<br>
$\mathcal{L}(x_{f}) = mse(\hat{x}_{f}, x_{f}) + \beta_{\phi} D_{KL}(\phi_{\theta}(\hat{s}_f|x_f)||\mathcal{N}(0, I))$<br>
$\mathcal{L}(y) = crossentropy(\hat{y}, y)$<br>
$\mathcal{L}_{Prior} = \mathcal{L}(y) + \mathcal{L}(x_{f})$<br>

<b>Prediction:</b><br>
$s'_{f} = \tau_{\alpha}(s'_{f}|x_s)$<br>
$y' = \psi_{\alpha}(s'_{f})$<br>

Loss Functions:<br>
$\mathcal{L}(s_{f}) = mse(s^{'}_{f, \mu}, \hat{s}_{f, \mu}) + \beta_{\tau} D_{KL}(\tau_{\alpha}(s'_{f}|x_{s})||\phi_{\theta}(\hat{s}_{f}|x_f))$<br>
$\mathcal{L}(y) = cross_entropy(y', y)$<br>
$\mathcal{L}_{Prediction} = \mathcal{L}(y) + \mathcal{L}(s_f)$<br>
