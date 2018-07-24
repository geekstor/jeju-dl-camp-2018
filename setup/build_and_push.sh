#!/usr/bin/env bash
set -x

export DOCKER_REGISTRY='retrocontestviebtzaqpdksjflr_1.gcr.io'
export docker_registry_username='xxx'
export docker_registry_password='xxx'

docker login $DOCKER_REGISTRY \
    --username $docker_registry_username \
    --password $docker_registry_password

#!/usr/bin/env bash
set -x

export dir=rainbow-agent

rm -rf $dir
mkdir $dir

cp ./rainbow.docker rainbow-agent
cp ../models/rainbow.py $dir/agent.py

export version='0511_npn_01350_110592000_totalstep3e6_lr_clip_1'

cd $dir
docker build -f rainbow.docker -t $DOCKER_REGISTRY/rainbow-agent:$version .

docker push $DOCKER_REGISTRY/rainbow-agent:$version
#gcloud container builds submit --timout 1h1m1s --tag gcr.io/dlcampjeju2018-207503/rl-image .
