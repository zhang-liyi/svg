# Stochastic Video Generation with a Recurrent Convolutional Variational Model

The project implements the following paper:
* Emily L. Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, 2018.

To run:
* Copy this repository locally.
* Inside `data` folder, unzip `mmnist.pickle.zip` and leave the unzipped file right there (under `data` folder).
* Open `svg.ipynb` and run the corresponding cells.

### Abstract

One approach to video frame prediction is deep probabilistic generative modeling. The idea is to model uncertainty in images through time by jointly using recurrence, convolution, and latent variables. We aim at implementing a variational recurrent model called stochastic video generation with learned prior (SVG-LP). We present a novel review of this problem from the probabilistic modeling perspective, and a novel software design in TensorFlow for implementing this model and its training and generation. Among features of this design is the ridding of for-loop across video frames, compared to the original PyTorch implementation. We partially reproduce the original paper’s results on stochastic moving MNIST.

```
./
├── README.md
├── checkpoints
│   ├── checkpoint
│   ├── model.data-00000-of-00001
│   └── model.index
├── data
│   └── mmnist.pickle.zip
├── E4040.2021Fall.VRNN.report.lz2574.pdf
├── figs
│   ├── loss.png
│   ├── lz2574_gcp_work_example_screenshot_1.png
│   ├── lz2574_gcp_work_example_screenshot_2.png
│   ├── lz2574_gcp_work_example_screenshot_3.png
│   └── ssim.png
├── files
│   ├── __pycache__
│   │   ├── model_svg.cpython-36.pyc
│   │   ├── models.cpython-36.pyc
│   │   └── utils.cpython-36.pyc
│   ├── model_svg.py
│   ├── models.py
│   └── utils.py
├── results
│   ├── losses.csv
│   ├── ssim.csv
│   ├── ssim_test.pickle
│   └── x_out.pickle
├── svg.ipynb
└── trained_models
    ├── model1
    │   ├── checkpoint
    │   ├── model.data-00000-of-00001
    │   └── model.index
    └── model3
        ├── checkpoint
        ├── model.data-00000-of-00001
        └── model.index

9 directories, 24 files
```
