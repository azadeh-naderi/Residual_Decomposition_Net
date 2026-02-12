## DecompNet

Image Decomposition with Masked CNN Autoencoders
DecompNet is a PyTorch framework for iterative image decomposition and reconstruction using multiple lightweight CNN autoencoders combined with fixed spatial Gaussian masks. The model decomposes an input image into spatially localized components and reconstructs it via a weighted sum of learned parts.
The framework supports both grayscale and RGB datasets, including ORL Faces, CIFAR-10/100, and ImageNet.

## Key Ideas

- Multi-component decomposition using N parallel CNN autoencoders
- Fixed spatial Gaussian masks applied to residuals before encoding
- Iterative residual sweeps for component refinement
- Closed-form ridge regression to solve component weights
- Works for 1-channel and 3-channel images with a unified codebase

## Supported Datasets

- ORL Faces (grayscale)
- CIFAR-10
- CIFAR-100
- ImageNet (ImageFolder format)
