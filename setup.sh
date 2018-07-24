#!/bin/bash
rm -rf $HOME/miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH" & source $HOME/miniconda/bin/activate 
echo "export PATH="$HOME/miniconda/bin:$PATH" & source $HOME/miniconda/bin/activate " >> ~/.bashrc
sleep 10s
source ~/.bashrc
sleep 10s
source ~/.bashrc
sleep 10s

sudo apt-get update
sudo apt install -y libgl1-mesa-glx
sudo apt-get install -y libxrender-dev
dpkg --add-architecture i386
sudo apt-get install -y lib32gcc1
sudo apt update && apt install -y libsm6 libxext6
sudo apt-get install -y python-opengl
sudo apt-get install -y xvfb
sudo apt-get install -y libav-tools
sudo apt-get install -y htop
sudo apt-get install -y moreutils
sudo apt-get update

sleep 5s

# TODO: Get Conda tensorflow-gpu linked to libcublas-9.0

sleep 10s
source ~/.bashrc
sleep 10s
source ~/.bashrc
sleep 10s

conda env create

source activate rl1

git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

python -m retro.import.sega_classics
