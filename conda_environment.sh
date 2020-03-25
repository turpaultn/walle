#!/bin/bash

conda create -n walle python=3.6
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
conda install pandas h5py
conda install pysoundfile librosa youtube-dl jupyterlab -c conda-forge

# ## Visualisation
conda install tsnecuda cuda101 -c cannylab  ## Only if gpu available
conda install seaborn

pip install sed-eval
pip install dcase_util

# If audioread NoBackendError:
conda install ffmpeg -c conda-forge

pip install --upgrade desed@git+https://github.com/turpaultn/DESED
