#!/bin/bash

source ~/ppgia/venv/bin/activate
./ee-processor-client.py --mq-username remote --mq-password remote --mq-hostname ibm-cuda --trained-network-file AlexNetWithExits_epoch_19_90.1_91.1.pth --network AlexNet --mq-queue alexnet --count 10
