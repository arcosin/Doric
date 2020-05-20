Doric Examples
==============

In this directory, there are three different Variational Auto Encoders, each implemented differently:

* `deform-cnn-vae.py`
* `vae-cnn.py`
* `vae-cnn-two-network.py`

`deform-cnn-vae.py` implements a VAE using Deformable convolutional networks in the encoder.

`vae-cnn.py` implements a basic convolutional VAE using a single column

`vae-cnn-two-network.py` implements the same convolutional VAE as `vae-cnn.py`, but splits the encoder and decoder into two different ProgNets, so that one can generate images from noise with the decoder


Running Examples
================
`deform-cnn-vae.py` provides several command line options:

| Flag         | Description                            | Default |
|--------------|----------------------------------------|---------|
| --cpu        | Whether the CPU should be used         | False   |
| --output     | Where to save samples after each epoch | output/ |
| --batch_size | Batch size                             | 100     |
| --epochs     | Epochs                                 | 50      |
| -lr          | Learning rate                          | 0.0005  |

Downloading CelebA dataset
==========================
* run `python3 download_celeba.py <dest.zip>` and extract to `../data/img_align_celeba/1`, where `1/` contains all the images