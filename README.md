# Image-Captioning-Chinese

This repository is my solution for ai challenger Image-Captioning(Chinese).

This is based on ruotianluo's [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)  repository. The modifications is:

## Requirements
Python 2.7 (because there is no [AI_Challenger-caption-eval](https://github.com/AIChallenger/AI_Challenger/tree/master/Evaluation/caption_eval) version for python 3) PyTorch 0.3 (along with torchvision)

You need to extract image feature for yourself, like fast-rcnn, resnet, etc.

## Train your own network on ai challenger
Download ai challenger dataset and extract.
First, download the ai challenger image caption data from [link](https://challenger.ai/competition/caption)

Second, extract image feature.

## Start training
```
python train.py
```
