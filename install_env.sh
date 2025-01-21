#!/bin/bash

conda create -n rlhf python=3.12
pip install flash-attn --no-build-isolation
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install vllm
pip install -e .