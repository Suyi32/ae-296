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

req_times = []

# send a small number of requests
for i in range(60):
    start_time = time.time()
    send_post_request()
    end_time = time.time()
    req_times.append(end_time - start_time)

print("Avg req time: ", sum(req_times)/len(req_times))

print("Total time cost: ", sum(req_times))
