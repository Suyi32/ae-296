import requests
import threading
import json
import time

def send_post_request():
    url = "http://127.0.0.1:8080/generate_stream"
    pload = {"inputs":"What is AI?", "parameters":{ 'max_new_tokens':32, 'frequency_penalty':1}, "lora_id": 0}   
    response = requests.post(url, data=json.dumps(pload), headers={'Content-type': 'application/json'})
    return response

# warmup 
print("warmup start")
send_post_request()
send_post_request()
send_post_request()
print("warmup done")

start_time = time.time()
# send a small number of requests
for i in range(10):
    send_post_request()
end_time = time.time()
print("time cost: ", end_time - start_time)
