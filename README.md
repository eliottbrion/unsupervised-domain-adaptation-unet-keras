# unsupervised-domain-adaptation-unet-keras

In this repository, we encourages the learning of domain-invariant features (i.e., the same features are learned whether the input is from a source domain like CTs or target domain such as Cone Beam CTs) through the use of a second network, the domain classifier, which interacts with the segmentation network in an adversarial fashion

Our proposed architecture unet\_L$x$ adapts u-net to unsupervised domain adaptation by backpropagation. The goal is to learn from labeled CTs features that are also useful for segmenting images from a different yet similar domain (the CBCTs). U-net is split in two parts: a feature extractor and a label predictor. The feature extractor learns features from the whole batch (both CTs and CBCTs), with it's final layer L$x$ aiming to (i) be useful for the label predictor and (ii) fool the domain classifier. The label predictor learns feature from half the batch in the layer L$x$ (the CTs) aiming to predict a segmentation mask for these CTs. The domain classifier uses these same features L$x$ to classify all images in the batch (both CTs and CBCTs) as being either CTs or CBCTs.

![alt text](unet_L9_cropped.PNG)

