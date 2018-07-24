curl -fsSL get.docker.com -o get-docker.sh
sh get-docker.sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

sudo apt-get install xvfb
sudo apt-get install libav-tools
sudo apt-get install python-opengl

sleep 10s
source ~/.bashrc
sleep 10s
conda env create

source activate simplesonic

git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

sudo docker pull openai/retro-env
sudo docker tag openai/retro-env remote-env

python -m retro.import.sega_classics
