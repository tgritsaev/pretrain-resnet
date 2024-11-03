# Homework 2, Large Scale Deep Learning 2024, CUB 

## Description

This repository is a solution for the 2-nd homework from the LSDL 2024 Course taught at the CUB. Throughout this howework I pretrained ResNet-18 following ["UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS"](https://arxiv.org/abs/1803.07728) and tested it on a classification task.

See the [**full statements**](https://github.com/isadrtdinov/lsdl-cub/blob/main/week02-pretext/homework.md) and the [**report**](https://wandb.ai/tgritsaev/Pretrain%20ResNet-18,%20HW-2.%20LSDL%202024,%20CUB./reports/Report-for-homework-2-at-Large-Scale-Deep-Learning-2024-course---Vmlldzo5NzY5NTc5?accessToken=z4oq1kq4ilpu439akuxvnuc0nk1wslz94mpt8pez461w37rq2gnyb0jg6l7ikdft).

## Reproducibility

* Setup environment:
```bash
conda create -n cv-pretrain python=3.10
conda activate cv-pretrain
pip install -r requirements.txt
```

* Download and unzip data, make sure you set up `kaggle.json` and joined the [competition](https://www.kaggle.com/competitions/lsdl-hw-2):
```bash
kaggle competitions download -c lsdl-hw-2
unzip lsdl-hw-2.zip
```

* Run the next to reproduce the final solution:
```bash
python pretrain.py
```
