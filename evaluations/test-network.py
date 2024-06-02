#!/usr/bin/env python

import uuid
import pickle
import logging
import torch
from datetime import datetime
import time
import json
import sys
import argparse
from datetime import datetime
sys.path.append('..')
from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits

parser = argparse.ArgumentParser(description='Network Tester')

parser.add_argument('--device', help='PyTorch device', default='cpu')
parser.add_argument('--trained-network-file', help='Trainet network file', required=True)
parser.add_argument('--network', help='Network to use AlexNet | MobileNet', required=True)
parser.add_argument('--input-size', help='Input size to the network', default=8)

args = parser.parse_args()

log_level = logging.INFO
logging.basicConfig(level=log_level,
                    format='%(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)

device = torch.device(args.device)
if args.network == 'MobileNet':
    model = MobileNetV2WithExits().to(device)
else:
    model = AlexNetWithExits().to(device)

model.load_state_dict(torch.load(args.trained_network_file, map_location=device))
model.eval()

x = torch.rand(1, 1, int(args.input_size), int(args.input_size)).to(device)
output = model(x)  # Run the network once to cache it
print(output)

e1 = model.forward_exit(0, x)
print(e1)

e2 = model.forward_exit(1, x)
print(e2)

# ./test-network.py --network MobileNet --trained-network-file ../trained_models/MobileNetV2WithExits_epoch_19_89.7_90.9.pth
# ./test-network.py --network AlexNet --trained-network-file ../trained_models/AlexNetWithExits_epoch_19_90.1_91.1.pth

