#!/bin/bash
version=12.4

conda create -n rlhf python=3.12
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=$version -c pytorch -c nvidia
conda install -c f"nvidia/label/cuda-{$version}.0" cuda-toolkit -y
pip install flash-attn --no-build-isolation
pip install vllm
pip install -e .

pip install sympy regex