# Homework 2, Large Scale Deep Learning 2024, CUB 

## Description

This repository is a solution for the 2-nd homework from the LSDL 2024 Course taught at CUB. Throughout this howework I pretrained ResNet-18 following ["UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS"](https://arxiv.org/abs/1803.07728) and tested it on a classification task.

**Full statements**: https://github.com/isadrtdinov/lsdl-cub/blob/main/week02-pretext/homework.md

**Report**:&emsp;&emsp;&emsp;&emsp;  TODO

## Reproducibility

* To download and unzip data, make sure you set up `kaggle.json` and joined the [competition](https://www.kaggle.com/competitions/lsdl-hw-2):
```bash
https://www.kaggle.com/competitions/lsdl-hw-2
unzip lsdl-hw-2.zip
```

* Setup environment:
```bash
conda create -n cv-pretrain python=3.10
conda activate cv-pretrain
pip install -r requirements.txt
```

* Run the next to reproduce the final solution:
```bash
python pretrain.py
```
