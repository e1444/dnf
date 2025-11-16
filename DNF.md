# Motivations
The motivating idea in this work is:
1. The data distribution for each class is generated from a simple latent distribution via some complex, invertible transformation.
2. The class-conditional latent distributions are separable.

As such, our goal is to learn an invertible mapping from the data space to a separable latent space. Learning an accurate mapping will allow us to:
1. Perform classification by evaluating class likelihoods in the latent space.
2. Get well-calibrated uncertainty estimates via class posterior probabilities.

## Related works
This work is inspired by several prior topics:
* **Normalizing flows:** The architecture borrows heavily from normalizing flows, which are invertible neural networks with tractable Jacobian determinants. This allows us to compute exact likelihoods.
* **Deep supervision:** The idea of applying losses at intermediate layers is inspired by deep supervision techniques, which have been shown to improve training stability and performance.
* **Hybrid generative-discriminative models:** The combination of discriminative and generative losses is inspired by hybrid models that aim to leverage the strengths of both approaches.

# Model Introduction
## Data
We have data $D \subseteq R^F$ with labels $c \in \{0, 1, ..., C-1\}$. Let $D_c$ be the subset of data points with class label $c$. $D_c$ has an unknown distribution with pdf $\pi_c$. **These are the true class distributions we wish to approximate.**

## Discriminative Normalizing Flow (DNF)
A DNF $f$ aims to map the input distribution to simple latent distributions $Z_c$ for each class, such that the class distributions are separable in the latent space. Let $Z_c$ have pdf $\varphi_c$.

The goal is to approximate the true data distribution $\pi_c$ with the following change of variables formula:
$$\pi_c(x) \approx \pi'_c(x) = \varphi_c(z) |\det J_{f}(x)|$$

where $z = f(x)$. The model will follow a neural network architecture with $L$ layers that is invertible:
$$f(x) = f_L \circ f_{L-1} \circ ... \circ f_1(x)$$

where each $f_j$ is an invertible module.

### Architecture
The DNF architecture follows the same design as normalizing flows. In our experiments, we borrow [RealNVP](https://arxiv.org/pdf/1605.08803) and [Glow](https://arxiv.org/pdf/1807.03039) architectures. The key property is that each layer is invertible with a tractable Jacobian determinant.

These architectures use components like affine coupling layers and invertible 1x1 convolutions to build complex, invertible transformations from simple, tractable building blocks.

### Training
Prior to initialization, we define fixed latent distributions for each class. The heuristic that we will follow is that we want to keep the latent distributions separated throughout all layers. In this sense, we don't want the network to "collapse" the distributions together at any point and tangle them.

As such, we will define have two parts to the loss function: the final loss function at the last layer, and auxiliary loss functions at each intermediate layer (deep supervision).

#### Deep supervision
Due to the unique architecture of normalizing flows, we can compute valid likelihoods at each intermediate layer. This allows us to apply any probabilistic loss function at each layer, not just the final output. Specifically, the distribution at layer $j$ is:
$$\pi'_{c,j}(x) = \varphi_c(z_j) |\det J_{f_j \circ ... \circ f_1}(x)|$$

where $z_j = f_j \circ ... \circ f_1(x)$. The deep supervision loss at layer $j$ is composed of 2 parts: 
* Entropy loss to encourage separability:
$$\mathcal{L}_{E,j}(x_i) = -\sum_c p_{c,j}(x_i) \log p_{c,j}(x_i)$$

    where $p_{c,j}(x_i) = \frac{\varphi_{c,j}(x_i)}{\sum_k \varphi_{k,j}(x_i)}$ is the posterior probability of class $c$ for sample $x_i$ at layer $j$.

    Minimizing this entropy encourages the model to produce a confident, 'peaky' posterior distribution for each sample, which serves as a proxy for pushing the latent representations into distinct, well-separated class regions.
* Cross-entropy loss to encourage correct classification.
$$\mathcal{L}_{CE,j}(x_i) = \frac{\varphi_{c_i,j}(x_i)}{\sum_c \varphi_{c,j}(x_i)}$$

Specifically, the total loss at layer $j$ is:
$$\mathcal{L}_j(x_i) = (1 - \beta_j)\mathcal{L}_{CE,j}(x_i) + \beta_j \mathcal{L}_{E,j}(x_i)$$

In our training procedure, we use a geometric schedule for $\beta_j$ such that:
$$\beta_j = \gamma^{L-j}$$

for some $\gamma \in (0, 1)$. This prioritizes that the model keeps the latent distributions separated, before gradually moving them to the correct locations

#### Final Loss
The loss at the final layer $L$ is composed of a cross-entropy loss and a negative log-likelihood loss:
$$\mathcal{L}_L(x_i) = (1 - \alpha)\mathcal{L}_{CE,L}(x_i) + \alpha \mathcal{L}_{NLL,L}(x_i)$$

In our training procedure, we use a relatively high value of $\alpha$ to encourage good likelihoods. The idea is that by the final layer, the latent distributions should already be well-separated, so we can focus on getting the correct shape and location.

### Inference
At inference time, we use the trained model to compute the posterior probability of each class for a new data point $x_0$.

1.  First, we compute the log-likelihood of $x_0$ under each class model, which are the same logits we computed during training:
    $$\log \pi'_c(x_0) = \log \varphi_c(f(x_0)) + \log |\det J_f(x_0)|$$

2.  We can then compute the posterior class probabilities by applying the softmax function to these log-likelihoods (which is equivalent to using Bayes' theorem with equal class priors):
    $$P(\text{class}=c | x_0) \approx \frac{\exp(\log \pi'_c(x_0))}{\sum_{j=0}^{C-1} \exp(\log \pi'_j(x_0))}$$

3.  The final prediction is the class with the highest posterior probability:
    $$\hat{c} = \arg\max_c P(\text{class}=c | x_0)$$

### Results
TBD

## Discussion
### Improvements over NGN
The NGN has several problems as discussed previously:
1. The Gaussian assumption on the class distributions is naive, and more importantly, degenerate. If we assume we know the true class distributions, we can achieve perfect classification performance without any learning.
    1. Importantly, estimating the covariance matrices is also an unstable operation, far more severe than computing log-likelihoods.
    2. Diagonal or spherical covariance assumptions similarly make the log-likelihoods trivial to compute.
2. The non-invertibility of the NGN mapping prevents us from computing class probabilities, leading to poor calibration. What we really end up learning is a scoring function, not a proper generative model.

These issues are unreconcilable and require fundamental rethinking of the model, leading to this iterative work. However, the spirit of the NGN is preserved:
1. We want to learn class distributions that are separable in some latent space.
2. We want to learn a singular mapping through the network.
3. We want to retain the analytical properties of the model.

### Limitations
1. Due to the properties of high-dimensional Gaussian distributions, the model has some limitations. An intuitive understanding is that in high dimensions, almost all the probability mass of a Gaussian is concentrated in a thin hypershell at a certain radius from the mean. While mapping to hypershells is feasible for complex networks (such as ours), the real crux of the problem is modelling uncertainty. An overlapping region of two high-dimensional Gaussians is typically a manifold of one lower dimension, and the overlapping region of three is two lower. This means that the model is implicitly biased towards making confident predictions, as the probability of being in the overlapping region of multiple classes is very low.

    In our experiments, we chose to initialize the latent class distributions such that they don't overlap; that there are only pairwise contacts. Although there are technically regions in which multiple classes have equal likelihoods, these regions are of very low measure, and the model is unlikely to map any data points there. This leads to our model only being able to predict a primary and secondary class with any confidence in practice; i.e., our model has minimum 50% confidence in all its predictions. This assumption is reasonable for datasets like MNIST, but may not hold in general.
2. Poor OOD detection. This is a limitation of most generative models. While the model may learn an accurate distribution over the training data, there is no guarantee that it will assign low likelihoods to OOD data. In practice, we find that our model often assigns high likelihoods to OOD samples, leading to overconfident predictions.
3. There are a huge amount of hyperparameters to tune, including architecture choices, latent distribution choices, and loss weights. A deep understanding of both the model and the data is required to make good choices. The model itself is also large and slow to train, making hyperparameter optimization difficult.

These issues are deeply rooted in the model architecture.