#!/bin/bash
sudo apt-get update
conda install -c conda-forge opencv=2.4
sudo apt install libgl1-mesa-glx
sudo apt-get install -y libxrender-dev
dpkg --add-architecture i386
sudo apt-get install lib32gcc1
sudo apt update && apt install -y libsm6 libxext6
sudo apt-get install xvfb
sudo apt-get install libav-tools
sudo apt-get install htop
sudo apt-get update
