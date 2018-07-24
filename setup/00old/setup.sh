curl -fsSL get.docker.com -o get-docker.sh
sh get-docker.sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

sudo apt-get install xvfb
sudo apt-get install libav-tools
sudo apt-get install python-opengl

source ~/.bashrc
git clone https://github.com/RishabGargeya/rl-seminar1.git
cd rl-seminar1

conda env create

source activate rl1

# xvfb-run -s python example.py

git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

docker pull openai/retro-env
docker tag openai/retro-env remote-env

python -m retro.import.sega_classics
