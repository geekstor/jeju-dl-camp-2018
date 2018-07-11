#!/bin/bash
sudo apt-get update
sudo apt install libgl1-mesa-glx
sudo apt-get install -y libxrender-dev
dpkg --add-architecture i386
sudo apt-get install lib32gcc1
sudo apt update && apt install -y libsm6 libxext6
sudo apt-get install xvfb
sudo apt-get install libav-tools
sudo apt-get install htop
sudo apt-get install moreutils
sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
sleep 10s
source ~/.bashrc
conda env create
python -m retro.import.sega_classics
