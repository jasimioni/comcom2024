#!/usr/bin/env python
import pika
import uuid
import pickle
import torch
from datetime import datetime
import time
import json
import sys
from datetime import datetime
sys.path.append('..')
from models.AlexNet import AlexNetWithExits

#mq_username = 'cuda'
#mq_password = 'cuda'
#mq_hostname = '192.168.32.23'
mq_username = 'remote'
mq_password = 'remote'
mq_hostname = '10.236.89.42' 
mq_queue = 'ee-processor'

device = 'cpu'
trained_network = 'AlexNetWithExits_epoch_19_90.1_91.1.pth'

device = torch.device(device)
model = AlexNetWithExits().to(device)
model.load_state_dict(torch.load(trained_network, map_location=device))
model.eval()

class EEProcessorClient(object):
    def __init__(self):
        credentials = pika.PlainCredentials(mq_username, mq_password)

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=mq_hostname, credentials=credentials))

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
            routing_key=mq_queue,
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
            return self.response

eeprocessor = EEProcessorClient()

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

sample = torch.rand(1, 1, 8, 8).to(device)

bb1 = model.backbone[0](sample)
pred_e1 = model.exits[0](bb1)

e2_start_time = time.time()
bb2 = model.backbone[1](bb1)
pred_e2 = model.exits[1](bb2)
e2_end_time = time.time()

e2_local_time = e2_end_time - e2_start_time

request = {
    'timestamp': now,
    'bb1': bb1,
    'sample': sample
}

print(f" [x] Requesting {now}")
print(pred_e2)

start = time.time()
response = eeprocessor.call(request)
end = time.time()
local_time = end - start

print(f"Processing bb2 + e2 locally would take: {1000 * e2_local_time:.3f} ms")

input_size = response['input_size']
output = response['output']
time_records = response['time_records']

# We are re-processing bb1 and e1 which are not needed, so dropping tha time from the ammount
should_desconsider_time = time_records['after_e1'] - time_records['after_device_map']

remote_time = time_records['end'] - time_records['start']

remote_time -= should_desconsider_time
local_time -= should_desconsider_time

network_latency = local_time - remote_time

print(f"Remotely processing took: started at {start} | ended at {end} | total: {1000 * local_time:.3f} ms")
print(f"Remote agent spent: started at {time_records['start']} | ended at {time_records['end']} | total: {1000 * remote_time:.3f} ms")
print(f"Network processing latency: {1000 * network_latency:.3f} ms")


print(f"Size: {input_size}")
print(output)

current = None
for key in time_records:
    if current is not None:
        elapsed = time_records[key] - current
        print(f"{key}: {1000 * elapsed:.3f} ms")
    current = time_records[key]