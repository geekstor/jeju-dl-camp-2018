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
sleep 10s
conda env create
sleep 10s
python -m retro.import.sega_classics








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
sudo apt install libgl1-mesa-glx
sudo apt-get install -y libxrender-dev
dpkg --add-architecture i386
sudo apt-get install lib32gcc1
sudo apt update && apt install -y libsm6 libxext6
sudo apt-get install python-opengl
sudo apt-get install xvfb
sudo apt-get install libav-tools
sudo apt-get install htop
sudo apt-get install moreutils
sudo apt-get update

sleep 5s

# TODO: Get Conda tensorflow-gpu linked to libcublas-9.0

sleep 10s
source ~/.bashrc
sleep 10s
source ~/.bashrc
sleep 10s

conda env create

source activate simplesonic

git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

python -m retro.import.sega_classics
