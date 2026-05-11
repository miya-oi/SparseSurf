docker rm -f foundation_stereo
DIR=$(pwd)/../
USER_HOME="${HOME:-/home/${USER}}"
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundation_stereo  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v "${USER_HOME}:${USER_HOME}" -v $DIR:$DIR -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE nvcr.io/nvidian/foundation_stereo:latest bash
