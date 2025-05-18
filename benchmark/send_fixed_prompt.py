import requests
import threading
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--token_num", type=int, default=1000)
args = parser.parse_args()
print("args: ", args)
def generate_prompt(token_num: int) -> str:
    return " ".join(["test"] * (token_num - 1))

def send_post_request(input_token_num: int):

    url = "http://127.0.0.1:8080/generate_stream"
    pload = {"inputs":generate_prompt(input_token_num), "parameters":{ 'max_new_tokens':32, 'frequency_penalty':1}, "lora_id": 0}   
    response = requests.post(url, data=json.dumps(pload), headers={'Content-type': 'application/json'})
    return response

# warmup 
print("warmup start")
send_post_request(2)
send_post_request(2)
send_post_request(2)
print("warmup done")

req_times = []

# send a small number of requests
for i in range(10):
    start_time = time.time()
    send_post_request(args.token_num)
    end_time = time.time()
    req_times.append(end_time - start_time)

print("Avg req time: ", sum(req_times)/len(req_times))

print("Total time cost: ", sum(req_times))
