git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

docker pull openai/retro-env
docker tag openai/retro-env remote-env

retro-contest run --agent $DOCKER_REGISTRY/simple-agent:v1 \
    --results-dir results --no-nv Airstriker-Genesis Level1

retro-contest run --agent $DOCKER_REGISTRY/simple-agent:v1 \
    --results-dir results --no-nv --use-host-data \
    SonicTheHedgehog-Genesis GreenHillZone.Act1 
