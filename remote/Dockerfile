# Adapted from
# https://github.com/rail-berkeley/doodad/blob/master/doodad/wrappers/easy_launch/example/docker_example/Dockerfile

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04


SHELL ["/bin/bash", "-c"]

##########################################################
### System dependencies
##########################################################

# Now let's download python 3 and all the dependencies
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    ffmpeg \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Not sure what this is fixing
COPY ./vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN useradd -m user

USER user
ENV HOME=/home/user
WORKDIR /home/user

# Not sure why this is needed
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

##########################################################
### MuJoCo
##########################################################
RUN mkdir -p $HOME/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d $HOME/.mujoco \
    && rm mujoco.zip
RUN mkdir -p $HOME/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d $HOME/.mujoco \
    && rm mujoco.zip
COPY ./vendor/mjkey.txt $HOME/.mujoco/mjkey.txt
RUN ln -s $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
ENV LD_LIBRARY_PATH $HOME/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH $HOME/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH $HOME/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}


##########################################################
### Example Python Installation
##########################################################
ENV PATH $HOME/miniconda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $HOME/miniconda && \
    rm /tmp/miniconda.sh

RUN conda update -y --name base conda && conda clean --all -y

RUN conda create -n jax-rl python=3.7
ENV OLDPATH $PATH
ENV PATH $HOME/miniconda/envs/jax-rl/bin:$PATH

RUN conda install -n jax-rl patchelf
RUN conda install -n jax-rl pip

RUN pip install --upgrade git+https://github.com/ikostrikov/jax-rl@main#egg
RUN pip install --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

WORKDIR /jax_rl