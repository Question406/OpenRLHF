#!/bin/bash
export NAME=rlhf
export version=12.4
conda create -n $NAME python=3.10 -y
conda activate $NAME
conda install pytorch torchvision torchaudio pytorch-cuda=$version -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-${version}.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX

pip install flash-attn --no-build-isolation
pip install vllm sympy regex
pip install -e .
