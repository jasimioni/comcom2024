#!/usr/bin/env python
from datetime import datetime
import sys
import time
import torch
import pickle
import pika
import socket
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
model(torch.rand(1, 1, 8, 8).to(device))  # Run the network once to cache it

credentials = pika.PlainCredentials(mq_username, mq_password)

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=mq_hostname, credentials=credentials))

channel = connection.channel()
channel.queue_declare(queue=mq_queue)


def on_request(ch, method, props, body):
    start = time.time()
    time_records = {'start': start}
    print(f" {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

    response = {}
    response['input_size'] = sys.getsizeof(body)
    body = pickle.loads(body)

    sample = body['sample'].to(device)
    bb1 = body['bb1'].to(device)

    time_records['after_device_map'] = time.time()

    bb1_rep = model.backbone[0](sample)
    time_records['after_bb1'] = time.time()

    e1 = model.exits[0](bb1)
    time_records['after_e1'] = time.time()

    bb2 = model.backbone[1](bb1)
    time_records['after_bb2'] = time.time()

    e2 = model.exits[1](bb2)
    time_records['after_e2'] = time.time()

    response['output'] = e2.to(torch.device('cpu'))
    time_records['end'] = time.time()

    response['time_records'] = time_records
    response['hostname'] = socket.gethostname()

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=pickle.dumps(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=mq_queue, on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
