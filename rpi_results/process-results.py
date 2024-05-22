#!/usr/bin/env python3

import json
import argparse
import os

parser = argparse.ArgumentParser(
    prog=os.path.basename(__file__),
    description="Process the results from ee-processor-client")

parser.add_argument('files',
                    help='Files to process',
                    type=argparse.FileType('r'),
                    nargs='+')

args = parser.parse_args()

output = []
for file in args.files:
    content = json.loads(file.read())
    output.append(content)

s_output = sorted(output, key = lambda x: (x['parameters']['network'],
                                           int(x['parameters']['input_size']), 
                                           int(x['parameters']['batch_size'])))

print('\t'.join([ 'network', 
                  'datacenter', 
                  'input_size', 
                  'batch_size', 
                  'where',
                  "e1_local_time",
                  "e1_process_time",
                  "e2_local_time",
                  "e2_process_time",
                  "request_local_time",
                  "request_process_time",
                  "e2_remote_time",
                  "network_latency",
                  "data_size"
                  ]))

for item in s_output:
    for where in [ 'local', 'remote' ]:
        print('\t'.join([
            item['parameters']['network'],
            item['parameters']['mq_hostname'],
            item['parameters']['input_size'], 
            item['parameters']['batch_size'],
            where,
            str(item['totals']['e1_local_time']),
            str(item['totals']['e1_process_time']),
            str(item['totals']['e2_local_time']),
            str(item['totals']['e2_process_time']),
            str(item['totals']['request_local_time']),
            str(item['totals']['request_process_time']),
            str(item['totals']['e2_remote_time']),
            str(item['totals']['network_latency']),
            str(item['totals']['input_size']),    
        ]))



