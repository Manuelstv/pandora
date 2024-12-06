#!/bin/sh
git config --global --add safe.directory /home/mstveras/mmdetection_2
git config --global user.email 'manuel.stveras@gmail.com'
git config --global user.name 'Manuelstv'

# Install Python packages
pip install -v -e .
pip install yapf==0.40.1
pip install future tensorboard

# Install line profiler
apt-get update
pip install line_profiler
pip install memory_profiler
pip instal h5py