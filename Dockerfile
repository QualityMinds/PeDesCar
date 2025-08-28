FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install python3.10 python3-pip nano git swig wget software-properties-common libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf -y
RUN apt-get install ffmpeg libsm6 libxext6 libglew-dev -y
RUN useradd rd -p "$(openssl passwd -1 SilverLaptop)" -s /bin/bash && mkdir /home/rd && chown rd:rd /home/rd
RUN apt-get upgrade -y

WORKDIR dataset-generation/
USER rd
ADD . ./
USER root

RUN pip install wheel setuptools pip --upgrade
RUN pip3 install --upgrade pip

# Mujoco installation
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar xvzf mujoco210-linux-x86_64.tar.gz
RUN mkdir -p /root/.mujoco/mujoco210/
RUN mv mujoco210/ /root/.mujoco/
COPY bin/mjkey.txt /root/.mujoco/mujoco210/mjkey.txt
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN chmod 777 -R bin/
RUN mkdir -p /ai-research/notebooks/testing_repos/

RUN chmod 777 -R pedestrian-impact/

RUN pip3 install -r ./requirements.txt
RUN pip3 install -e pedestrian-impact/ # to install the environments


RUN git clone https://github.com/robfiras/ls-iq.git
RUN python3 -m pip install -e ls-iq/

RUN git clone --branch v0.1.0 https://github.com/robfiras/loco-mujoco.git
RUN python3 -m pip install loco-mujoco/
RUN loco-mujoco-download

