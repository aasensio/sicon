# Stokes inversion based on convolutional neural networks

[![github](https://img.shields.io/badge/GitHub-aasensio%2Fsicon-blue.svg?style=flat)](https://github.com/aasensio/sicon)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/aasensio/sicon/blob/master/LICENSE)
[![ADS](https://img.shields.io/badge/ADS-arXiv190403714A-red.svg)](https://ui.adsabs.harvard.edu/abs/2019arXiv190403714A/abstract)
[![arxiv](https://img.shields.io/badge/arxiv-1904.03714-orange.svg?style=flat)](https://arxiv.org/abs/1904.03714)

## Introduction

Spectropolarimetric inversions are routinely used in the field of Solar Physics 
for the extraction of physical information from observations. The application 
to two-dimensional fields of view often requires the parallelization of the 
inversion codes. Still, the time spent on the process is very large.
SICON (standing for Stokes Inversion based on COnvolutional Neural networks) is a new 
inversion code based on the application of convolutional neural 
networks that can provide the thermodynamical and magnetic properties 
of two-dimensional maps instantaneously.
We use two different architectures using fully convolutional neural 
networks. We use synthetic Stokes profiles obtained from two numerical 
simulations of different structures of the solar atmosphere for training.

This repository is the release version of SICON. We provide both
training and evaluation scripts.


## Datasets

If you want to retrain any of the two architectures you will need to download
the training data from https://owncloud.iac.es/index.php/s/bGkJrn8mCuB4j9n

For evaluation, you can use your own Hinode dataset. However, we provide 
a patch of the active region AR10933 that can be downloaded from the same
server.

## Encoder-decoder architecture

### Dependencies

    numpy
    h5py
    scipy
    tqdm
    nvidia_smi
    pytorch

Within an Anaconda environment, standard packages can be installed with

    conda install numpy h5py scipy tqdm

`pytorch` can be installed following the instructions on http://pytorch.org

`nvidia_smi` can be installed using the recipe from fastai

    conda install nvidia-ml-py3 -c fastai

### Training


Training the encoder-decoder architectures simply requires to run the training as:

    python train_encdec.py

If you have an NVIDIA GPU on your system, it will make use of it to accelerate
the computations. If not, prepare to wait for a long time. Typical running times
in a P100 GPU are 90 seconds per epoch. A training with 50 epochs will last
for slightly longer than an hour.


## Evaluation

The evaluation of this architecture can be easily done by typing:

    python evaluate_encdec.py

It will generate figures and an HDF5 file with the output. You can easily
modify this script to read your own dataset. 


## Concatenate architecture

### Dependencies

    numpy
    h5py
    tensorflow
    keras

Within an Anaconda environment, depending if you have a GPU or not the command
will be different:
    
    conda install tensorflow keras

    conda install tensorflow-gpu 

### Training

Training the encoder-decoder architectures simply requires to run the training as:

    python train_concat.py


### Evaluation
The evaluation of this architecture can be easily done by typing:

    python evaluate_concat.py

It will generate figures and an numpy file with the output.
