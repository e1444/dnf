# Motivations
The motivating idea in this work is:
1. The data distribution for each class is generated from a simple latent distribution via some complex, invertible transformation.
2. We can learn a transformation and latent distributions such that the class distributions are separable in the latent space.

As such, if both our transformation and latent distributions are learned well, we can achieve two things:
1. Good classification performance, since the class distributions are separable in the latent space.
2. Well-calibrated class probabilities, since we can compute exact likelihoods using the change of variables formula.
3. OOD detection, since we can identify inputs that have low likelihood under all class distributions.

## Improvements over NGN
The NGN has several problems as discussed previously:
1. The Gaussian assumption on the class distributions is naive, and more importantly, degenerate. If we assume we know the true class distributions, we can achieve perfect classification performance without any learning.
2. The non-invertibility of the NGN mapping prevents us from computing class probabilities, leading to poor calibration. What we really end up learning is a scoring function, not a proper generative model.

These issues are unreconcilable and require fundamental rethinking of the model, leading to this iterative work. However, the spirit of the NGN is preserved:
1. We want to learn class distributions that are separable in some latent space.
2. We want to learn a singular mapping through the network.

## Related works
This work is inspired largely by Normalizing Flows (NFs), which are invertible neural networks that can compute exact likelihoods using the change of variables formula. NFs have been used primarily for generative modelling. The typical classification approach with NFs is to train one NF per class distribution, and then perform inference using Bayes' theorem. However, this approach does not encourage class separation in the latent space, leading to suboptimal classification performance.

This work borrows architecture from NFs, but presents a new training paradigm that focus on separability in the latent space.

# Model Introduction
## Data
We have data $D \subseteq R^F$ with labels $c \in \{0, 1, ..., C-1\}$. Let $D_c$ be the subset of data points with class label $c$. $D_c$ has an unknown distribution with pdf $\pi_c$.

## Discriminative Normalizing Flow (DNF)
A DNF $f$ aims to map the input distribution to simple latent distributions $Z_c$ for each class, such that the class distributions are separable in the latent space. Let $Z_c$ have pdf $\varphi_c$.

The goal is to approximate the true data distribution $\pi_c$ with the following change of variables formula:
$$\pi'_c(x) = \varphi_c(z) |\det J_{f}(x)|$$

where $z = f(x)$. This presents several immediate challenges:
1. We need to choose a suitable architecture for $f$ that is invertible and has a tractable Jacobian determinant.
2. We need to choose suitable latent distributions $\varphi_c$ that are simple yet separable.
3. We need to design a training procedure to learn both $f$ and the latent distributions.

Point 1 is addressed by borrowing architectures from normalizing flows, which are designed to be invertible with tractable Jacobians. Points 2 and 3 are addressed in the following sections.

### Architecture
A DNF is an invertible network that maps with a singular path through the network. The model consists of modules consisting of:
1. ActNorm layers: These are essentially invertible batch normalization layers that scale and shift the activations.
2. Coupling layers: These are layers that split the input into two parts, transform one part conditioned on the other, and then recombine them. This ensures invertibility and a tractable Jacobian.
3. Permutation layers: These layers permute the dimensions of the input to ensure that all dimensions are transformed across multiple coupling layers.

### Training
This model has two loss components: a discriminative loss and a generative loss. We want the latent space distributions to be both separable and high-likelihood. While these two objectives are not at odds, degenerate solutions exists for both objectives, which do compete with each other.

#### Discriminative Loss
We want the latent distributions to be separable. To achieve this, we will use a cross-entropy loss on the class likelihoods. This can be thought of as such: at each training sample, we want the likelihood of the correct class to be high, and the likelihoods of the incorrect classes to be low. If our samples span the entire space, this will force the class distributions to be separated.

We will denote the discriminative loss $\mathcal{L}_D$. For a training sample $x_i$ with true class label $c_i$:
1.  First, we compute the **logit** for **every** class $c \in \{0, ..., C-1\}$. The logit is the log-likelihood of the data point $x_i$ under the model for class $c$:
    $$\text{logit}_c(x_i) = \log \pi'_c(x_i) = \log \varphi_c(f(x_i)) + \log |\det J_f(x_i)|$$

2.  The loss is the standard cross-entropy loss on these logits with respect to the true class $c_i$:
    $$\mathcal{L}_D(x_i) = -\log \left( \frac{\exp(\text{logit}_{c_i}(x_i))}{\sum_{j=0}^{C-1} \exp(\text{logit}_j(x_i))} \right)$$

#### Generative Loss
While learning a separable latent distribution is good for classification, we also require that the our model accurately approximates the posterior distribution of the data. To achieve this, we use a negative log-likelihood loss that encourages high likelihoods for the data points under their respective class models.

We will denote the generative loss $\mathcal{L}_G$. For a training sample $x_i$ with true class label $c_i$:
$$\mathcal{L}_G(x_i) = -\log \pi'_{c_i}(x_i) = -\log \varphi_{c_i}(f(x_i)) - \log |\det J_f(x_i)|$$

#### Combined Loss
The combined loss is a weighted combination of the discriminative and generative losses:
$$\mathcal{L}(x_i) = \mathcal{L}_D(x_i) + \alpha \mathcal{L}_G(x_i)$$
where $\alpha$ is a hyperparameter that controls the trade-off between classification performance and generative performance.

#### Deep supervision
To encourage better learning throughout the network, we can apply deep supervision by adding auxiliary losses at intermediate layers of the network. This involves applying the total loss function at several points in the network, not just the final output. This can help the model learn more robust features and improve convergence.

Notably, the normalizing flow architecture is uniquely suited for deep supervision, since each intermediate layer represents a real distribution, giving meaningful likelihoods at each layer.

Let our model have $L$ layers, such that we can denote the model as:
$$f(x) = f_L \circ f_{L-1} \circ ... \circ f_1(x)$$

Then, we know that the output of layer $j$ is:
$$z_j = f_j \circ f_{j-1} \circ ... \circ f_1(x)$$

which represents a valid distribution with pdf:
$$\pi'_{c,j}(x) = \varphi_c(z_j) |\det J_{f_j \circ ... \circ f_1}(x)|$$

where $z_j = f_j \circ ... \circ f_1(x)$. This is tractable to compute, and provides a way to apply the total loss at layer $j$:
$$\mathcal{L}_j(x_i) = \mathcal{L}_{D,j}(x_i) + \alpha \mathcal{L}_{G,j}(x_i)$$

The final loss is then the sum of the losses at each layer:
$$\mathcal{L}_{total}(x_i) = \sum_{j=1}^{L} \beta_j\mathcal{L}_j(x_i)$$

This also opens up the possibility of using per-layer $\alpha_j$ values, allowing us to weight the discriminative and generative losses differently at each layer. There is some consideration for this approach, since earlier layers may benefit more from discriminative loss to encourage separability, while later layers may benefit more from generative loss to ensure accurate likelihoods.

##### Choosing beta
With this formulation, we have many more hyperparameters to choose, and it is non-obvious what is the best formulation. One approach is to draw a parallel to ResNets: we are effectively creating "gradient highways" through the network by applying losses at intermediate layers. This would suggest an exponential decay schedule, where the final layer has weight 1 ($\beta_L=1$), and earlier layers have exponentially decreasing weights (e.g., $\beta_j = \gamma^{L-j}$ for some $\gamma < 1$). This prioritizes the final output while still providing useful gradients throughout the network.

*Note: Why not just use residual connections? The key problem is that residual connections are not generally invertible, and require constraints on the coupling layers to ensure invertibility. This is an active area of research called "residual flows"*

#### Training Procedure
The main challenge in training is to avoid degenerate solutions. For example, if the latent distributions collapse to a single point, the model can achieve perfect classification but poor likelihoods. Conversely, if the latent distributions overlap significantly, the model can achieve high likelihoods but poor classification.

To mitigate this, we will train in two phases:
1. Train with frozen latent distribution parameters first to learn a good mapping. This allows the model to focus on learning separable latent distributions without worrying about degeneracy.
2. Unfreeze latent distributions parameters and continue training with both losses. Increase $\alpha$ gradually to encourage better likelihoods.

### Inference
At inference time, we use the trained model to compute the posterior probability of each class for a new data point $x_0$.

1.  First, we compute the log-likelihood of $x_0$ under each class model, which are the same logits we computed during training:
    $$\log \pi'_c(x_0) = \log \varphi_c(f(x_0)) + \log |\det J_f(x_0)|$$

2.  We can then compute the posterior class probabilities by applying the softmax function to these log-likelihoods (which is equivalent to using Bayes' theorem with equal class priors):
    $$P(\text{class}=c | x_0) \approx \frac{\exp(\log \pi'_c(x_0))}{\sum_{j=0}^{C-1} \exp(\log \pi'_j(x_0))}$$

3.  The final prediction is the class with the highest posterior probability:
    $$\hat{c} = \arg\max_c P(\text{class}=c | x_0)$$

### Limitations
1. There are a huge amount of hyperparameters to tune, including architecture choices, latent distribution choices, and loss weights. A deep understanding of both the model and the data is required to make good choices.