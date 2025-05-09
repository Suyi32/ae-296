import requests
import threading
import json
import time

def send_post_request():
    url = "http://127.0.0.1:8080/generate_stream"
    pload = {"inputs":"What is AI?", "parameters":{ 'max_new_tokens':32, 'frequency_penalty':1}, "lora_id": 0}   
    response = requests.post(url, data=json.dumps(pload), headers={'Content-type': 'application/json'})
    return response

# Create two threads to send the POST requests concurrently
threads = []
for i in range(1):
    threads.append( threading.Thread(target=send_post_request, args=()) )

# Start the threads
for thread in threads:
    thread.start()

# Wait for the threads to finish and collect the responses
for thread in threads:
    thread.join()