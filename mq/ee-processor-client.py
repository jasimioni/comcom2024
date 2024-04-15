#!/usr/bin/env python
import pika
import uuid
import pickle
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

parser = argparse.ArgumentParser(description='Early Exits processor client.')

parser.add_argument('--mq-username', help='RabbitMQ username')
parser.add_argument('--mq-password', help='RabbitMQ password')
parser.add_argument('--mq-hostname', help='RabbitMQ hostname', required=True)
parser.add_argument('--mq-queue', help='RabbitMQ queue', default='ee-processor')
parser.add_argument('--device', help='PyTorch device', default='cpu')
parser.add_argument('--trained-network-file', help='Trainet network file', required=True)
parser.add_argument('--network', help='Network to use AlexNet | MobileNet', required=True)
parser.add_argument('--count', help='Number of tests to run', default=1)
parser.add_argument('--input-size', help='Input size to the network', default=8)

args = parser.parse_args()

device = torch.device(args.device)
if args.network == 'MobileNet':
    model = MobileNetV2WithExits().to(device)
else:
    model = AlexNetWithExits().to(device)

model.load_state_dict(torch.load(args.trained_network_file, map_location=device))
model.eval()
model(torch.rand(1, 1, int(args.input_size), int(args.input_size)).to(device))  # Run the network once to cache it

class EEProcessorClient(object):
    def __init__(self):
        connection_params = { 'host': args.mq_hostname }
        if args.mq_username and args.mq_password:
            credentials = pika.PlainCredentials(args.mq_username, args.mq_password)
            connection_params['credentials'] = credentials

        self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(**connection_params))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, body):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=args.mq_queue,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=pickle.dumps(body))
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        
        try:
            return pickle.loads(self.response)
        except Exception as e:
            raise Exception(f"Failed to load: {e}")

eeprocessor = EEProcessorClient()

totals = {
    'e1_local_time': 0,
    'e2_local_time': 0,
    'e2_process_local_time': 0,
    'request_local_time': 0,
    'request_process_time': 0,
    'e2_remote_time': 0,
    'network_latency': 0,
    'input_size': 0
}
for i in range(int(args.count)):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    sample = torch.rand(1, 1, int(args.input_size), int(args.input_size)).to(device)

    e1_start_time = time.time()
    bb1 = model.backbone[0](sample)
    pred_e1 = model.exits[0](bb1)
    e1_end_time = time.time()

    e1_local_time = e1_end_time - e1_start_time

    e2_start_time = time.time()
    e2_process_start_time = time.process_time()
    bb2 = model.backbone[1](bb1)
    pred_e2 = model.exits[1](bb2)
    e2_process_end_time = time.process_time()
    e2_end_time = time.time()

    e2_local_time = e2_end_time - e2_start_time
    e2_process_local_time = e2_process_end_time - e2_process_start_time

    request = {
        'timestamp': now,
        'bb1': bb1,
        'sample': sample
    }

    print(f" [x] Requesting {now}")
    print(pred_e2)

    start = time.time()
    process_start = time.process_time()
    response = eeprocessor.call(request)
    process_end = time.process_time()
    end = time.time()
    request_local_time = end - start
    request_process_time = process_end - process_start

    input_size = response['input_size']
    output = response['output']
    time_records = response['time_records']
    hostname = response['hostname']

    # Remotely, we are re-processing bb1 and e1 which are not needed, so removing that time from the equation
    should_desconsider_time = time_records['after_e1'] - time_records['after_device_map']

    remote_time = time_records['end'] - time_records['start']

    remote_time -= should_desconsider_time
    request_local_time -= should_desconsider_time

    network_latency = request_local_time - remote_time
    
    statistics = {
        'hostname': hostname,
        'e1_local_time': e1_local_time * 1000,
        'e2_local_time': e2_local_time * 1000,
        'e2_process_local_time': e2_process_local_time * 1000,
        'request_local_time': request_local_time * 1000,
        'request_process_time': request_process_time * 1000,
        'e2_remote_time': remote_time * 1000,
        'network_latency': network_latency * 1000,
        'input_size': input_size
    }

    for key in totals:
        totals[key] += statistics[key]

    print(json.dumps(statistics, indent=2))

    print(f"Processed by {hostname}")
    print(f"Processing bb1 + e1 locally took {1000 * e1_local_time:.3f}")
    print(f"Processing bb2 + e2 locally would take: {1000 * e2_local_time:.3f} ms")
    print(f"Remotely processing took: started at {start} | ended at {end} | total: {1000 * request_local_time:.3f} ms")
    print(f"Remote agent spent: started at {time_records['start']} | ended at {time_records['end']} | total: {1000 * remote_time:.3f} ms")
    print(f"Network processing latency: {1000 * network_latency:.3f} ms")
    print(f"Size: {input_size}")

    current = None
    for key in time_records:
        if current is not None:
            elapsed = time_records[key] - current
            print(f"{key}: {1000 * elapsed:.3f} ms")
        current = time_records[key]

for key in totals:
    print(f"Average {key}: {totals[key] / int(args.count)}")