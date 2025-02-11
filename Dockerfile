### Base container ###
# Select the proper image version along with
# the tensorflow version defined in pyproject.toml.
# https://www.tensorflow.org/install/source?hl=ja#gpu_support_2
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS base
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /kaggle-imaterialist2020-model

### Install Python ###
FROM base AS py-install
RUN apt-get update && apt-get install -y \
  curl \
  make \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  llvm \
  libncurses5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev
ENV PYTHON_VERSION=3.7.10
RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
  tar zxvf Python-${PYTHON_VERSION}.tgz && cd Python-${PYTHON_VERSION} && ./configure && make && make install && \
  cd ../ && rm -r Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz
# Update pip to quickly install grpc libraries
RUN pip3 install -U pip
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python \
  && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip
RUN apt-get -y install git

### Install the app ###
FROM py-install AS app-install

# Build python libraries
RUN pip3 install poetry==1.1.8
COPY kaggle_imaterialist2020_model/ ./kaggle_imaterialist2020_model/
COPY tf_tpu_models/ ./tf_tpu_models/
COPY tools/ ./tools/
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
  # To install libclang (12.0.0)
  # which Poetry can't install while Pip can.
  # https://github.com/python-poetry/poetry/issues/3591#issuecomment-765840042
  && pip install .
