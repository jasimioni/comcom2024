#!/bin/bash

./evaluate-network.py --model mobilenet --trainedmodel ../trained_models/MobileNetV2WithExits_epoch_19_89.7_90.9.pth \
                      --savefolder original/MobileNetV2WithExits/ --dataset-folder ../MOORE/ --batch-size 250

./evaluate-network.py --model alexnet --trainedmodel ../trained_models/AlexNetWithExits_epoch_19_90.1_91.1.pth \
                      --savefolder original/AlexNetWithExits/ --dataset-folder ../MOORE/
