# Stochastic Video Generation with a Variational Recurrent Convolutional Model

The project implements the following paper:
* Emily L. Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, 2018.

To run:
* Copy this repository locally.
* Inside `data` folder, unzip `mmnist.pickle.zip` and leave the unzipped file right there (under `data` folder).
* Open `svg.ipynb` and run the corresponding cells.

### Introduction

One approach to video frame prediction is deep probabilistic generative modeling. The idea is to model uncertainty in images through time by jointly using recurrence, convolution, and latent variables. I aim at implementing a variational recurrent convolutional model called stochastic video generation with learned prior (SVG-LP). I include here an original program for this model in TensorFlow, and partially reproduce the original paper’s results on stochastic moving MNIST.

