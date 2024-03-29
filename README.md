# unsupervised-domain-adaptation-unet-keras

If you use the code, please cite the paper:
```
@article{brion2021domain,
  title={Domain adversarial networks and intensity-based data augmentation for male pelvic organ segmentation in cone beam CT},
  author={Brion, Eliott and L{\'e}ger, Jean and Barrag{\'a}n-Montero, AM and Meert, Nicolas and Lee, John A and Macq, Benoit},
  journal={Computers in Biology and Medicine},
  pages={104269},
  year={2021},
  publisher={Elsevier}
}
```

In this repository, we use u-net for male pelvic organ segmentation in an unsupervised domain adaptation setting. The code is written with the Keras package.

The proposed architecture has three parts: a feature extractor (FE) $G_f(\cdot, \theta_f)$, a label predictor (LP) $G_y(\cdot, \theta_y)$ and a domain classifier (DC) $G_d(\cdot, \theta_d)$, with $\theta_f$, $\theta_y$ and $\theta_d$ the weights of the respective networks (see figure bellow). The two first parts are similar to u-net. As for classical u-net, the FE takes as input a batch of images and the LP uses these features to output predicted segmentations (noted $\hat{y}_i$ for the $i$th example). Still as in classical u-net, the FE learns features from the whole batch, and its last features are noted L$z$ ($z=1,2,...,11$ is a hyper-parameter). The LP, however, only uses the features from the first half of the batch. Indeed we choose to organize the batch with the first half containing only source domain images (in our application, CTs) and the second half containing only target domain images (CBCTs). The features L$z$ corresponding to the first half of the batch are used by the LP to predict a segmentation for bladder, rectum and prostate. All features in L$z$ (i.e., from both halves of the batch) are used by the third part of the architecture: the DC. For each image in the batch, the DC outputs the probability $\hat{d}_i$ for the $i$th example to belong to the target domain. The DC interacts with the FE in an adversarial fashion, encouraging it to learn domain-invariant features (i.e., the same features are learned whether the input is from a source domain or target domain).

Following [1] and [2], this is achieved by finding the saddle point of the following loss function:

$$
    L_{tot}(\theta_f,\theta_y, \theta_d) = \sum_{\substack{i=1..N \\ d_i=0}} L_{seg}(\hat{y}_i(\theta_f, \theta_y), y_i), d_i)
$$

$$
+\sum_{i=1..N} L_{dom}(\hat{d}_i(\theta_f, \theta_d), d_i)
$$

where $N$ is the number of example for each domain (i.e., there are $2N$ examples in total). In this loss, $L_{seg}$ is the dice loss and encourage both the FE and the LP to learn features useful for source domain image segmentation. The expression $L_{dom}$ is the cross-entropy and encourages (i) for the FE, the learning of domain-invariant features, (ii) for the DC, the learning of features allowing to predict whether the features in L$z$ are activated by an input source domain image or target domain image.

The label predictor output is:

$$
    \hat{y}_i(\theta_f, \theta_y) = G_y((G_f(\mathbf{x}_i; \theta_f); \theta_y)
$$

To estimate the saddle point of $L(\theta_f, \theta_y, \theta_d)$ by backpropagation and to control the strengh of the interaction between the FE and the DC, we follow [2] and use the *gradient reversal layer*, defined as $R(\mathbf{x}) = \mathbf{x}$ and $\mathrm{d}R(\mathbf{x})/\mathrm{d}\mathbf{x} = -\lambda \mathbf{I}$. The larger $\lambda$, the higher the interaction between DC and FE. The DC's output then writes:

$$
    \hat{d}_i(\theta_f, \theta_y) = G_d(R(G_f(x_i; \theta_f)); \theta_y)
$$

where $x_i$ is the $i$th input image. In practise, setting $\lambda$ to a fixed value throughout learning leads to instabilities. To address this issue, we set $\lambda$ to zero during the first $e_0$ epochs (i.e., the FE and the DC learn their task independently) and then increase $\lambda$ linearly until it reaches the value $\lambda_{max}$ after the total number of epochs $n_{epochs}$:

$$
    \lambda(e) = \max \Big(0, \lambda_{max} \frac{e-e_0}{n_{epochs}-e_0}\Big),
$$
where $e$ is the current epoch. In this expresssion, $\lambda_{max}$, $e_0$ and $n_{epochs}$ are hyperparameters.

![alt text](unet_L9_cropped.PNG)

# References
[1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

[2] Ganin, Y., & Lempitsky, V. (2014). Unsupervised domain adaptation by backpropagation. arXiv preprint arXiv:1409.7495.
