## Build docker image
sudo docker build -f Dockerfile -t od - < Dockerfile

## Run docker image
sudo docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 8888:8888 -v $(pwd):/volume: od:latest

## Cleanup
sudo docker image prune
sudo docker builder prune