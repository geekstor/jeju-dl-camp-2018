curl -fsSL get.docker.com -o get-docker.sh
sh get-docker.sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

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

sleep 10s
source ~/.bashrc
sleep 10s

wget https://raw.githubusercontent.com/fastai/courses/master/setup/install-gpu.sh
sudo sh install-gpu.sh


sleep 10s
conda env create

source activate simplesonic

git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

sudo docker pull openai/retro-env
sudo docker tag openai/retro-env remote-env

python -m retro.import.sega_classics
