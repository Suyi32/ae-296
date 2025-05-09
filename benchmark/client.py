import asyncio
import aiohttp
import json
import argparse
import random
from datetime import datetime
import time
import numpy as np
from transformers import LlamaTokenizer


url = "http://127.0.0.1:8080/generate_stream"
headers = {'Content-Type': 'application/json'}
# tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = LlamaTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16")
word_tokenize = lambda x: tokenizer(x).input_ids


def parse_trace(trace_dir):
    event_times = []
    content_start = False
    with open(trace_dir, "r") as fr:
        for line in fr:
            if "=======" in line:
                content_start = True
                continue
            if content_start:
                event_times.append( float(line.strip()) )
    return event_times


def gen_pload():
    # f = open('dataset/alpaca_data.json')
    f = open('/workspace/aaas-data/dataset/lmsys_data.json')
    data = json.load(f)

    # set random seed, then sameple from teh dataset.
    random.seed(0)
    sample = random.sample(data, k=1000)
    
    while True:
        for entry_id, entry in enumerate(sample):
            x = entry["instruction"] + " " + entry["input"]
            res = {"inputs": x, 
                "parameters":{ 'max_new_tokens': len(word_tokenize(entry["output"])), 'frequency_penalty':1, 'do_sample': False, 'ignore_eos': True}, 
                'model_dir': 'huggyllama/llama-7b',
                'lora_dir': 'dummy-lora-7b-rank-8-{}'.format(entry_id%32), 
                "lora_id": 0}
            
            yield res


async def send_one_request(pload):
    request_start_time = time.time()
    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            print(pload)
            async with session.post(url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                    chunks.append(chunk)

            output = b"".join(chunks).decode("utf-8")
            # output = json.loads(output.replace("'", "\""))
            # print(output)
            # print(len(chunks))
            
            # if '\"finished\": -1' not in output:
            #     break
            if '\"finished\": true' in output:
                break
            else:
                first_token_latency = None
                break
    request_end_time = time.time()
    request_latency = request_end_time - request_start_time

    return (request_latency, len(word_tokenize(pload["inputs"])), len(chunks), first_token_latency)


async def send_requests( reqs_num, interval, load):
    async with aiohttp.ClientSession() as session:
        tasks = []
        counter = 0
        while counter < reqs_num:
            counter += 1
            pload = load.__next__()
            task1 = asyncio.create_task(send_one_request(session, pload))
            tasks.append(task1)
            await asyncio.sleep( interval )
            print("Sent request #: {}. At {}".format(counter, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
        await asyncio.gather(*tasks)
        tasks.clear()


async def replay_traces(event_times, load):
    generated_tokens = 0
    replay_start = time.time()
    tasks = []
    for event_id, event_time in enumerate(event_times):
        if event_id == 0:
            await asyncio.sleep( event_time )
        else:  
            await asyncio.sleep( event_times[event_id] - event_times[event_id-1] )

        pload = load.__next__()
        generated_tokens += pload["parameters"]["max_new_tokens"]
        task = asyncio.create_task(send_one_request(pload))
        tasks.append(task)

        print("Sent request #: {}. At {}".format(event_id, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
    resp_info = await asyncio.gather(*tasks)
    replay_end = time.time()
    ##
    # print(latency)
    resp_lats = [ item[0] for item in resp_info ]
    promt_tok_length = [ item[1] for item in resp_info ]
    resp_tok_length = [ item[2] for item in resp_info ]
    first_tok_lat = [ item[3] for item in resp_info ]

    print("Num of Reqs: {}".format(len(resp_lats)))
    print("Num of Resps: {}".format(len(resp_tok_length)))
    print("Total Benchmarking Latency: {:.3f} s".format( (replay_end - replay_start) ))
    avg_per_token_latency = np.mean([
        resp_lats[i] / (promt_tok_length[i] + resp_tok_length[i])
        for i in range(len(resp_lats))
    ])
    print("Average lat per token: {:.3f} s".format(avg_per_token_latency))
    avg_per_output_token_latency = np.mean([
        resp_lats[i] / resp_tok_length[i]
        for i in range(len(resp_lats))
    ])
    print("Average lat per output token: {:.3f} s".format( avg_per_output_token_latency )) 
    print("Avg Latency Per Req: {:.6f} s".format( np.mean(resp_lats) ))
    print("Total number of generated tokens: {} (dataset); {} (resp)".format(generated_tokens, np.sum(resp_tok_length)))

    avg_first_token_latency = np.mean(first_tok_lat)
    print(f"Average first token latency: {avg_first_token_latency:.3f} s")

    print("resp_lats: {}".format(resp_lats))
    print("resp_tok_length: {}".format(resp_tok_length))
    print("promt_tok_length: {}".format(promt_tok_length))
    print("first_tok_lat: {}".format(first_tok_lat))

    return
    # tasks.clear()


if __name__ == '__main__':
    load = gen_pload()
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=1, help='duration of the test in seconds')
    parser.add_argument('--interval', type=float, default=0.3, help='interval between requests in seconds')
    parser.add_argument('--trace_dir', type=str, default=None, help='interval between requests in seconds')
    args = parser.parse_args()

    if args.trace_dir == None:
        # Start the event loop and run the coroutine indefinitely
        reqs_num = int(args.duration / args.interval)
        asyncio.run( send_requests(reqs_num, args.interval, load) )
    else:
        # read trace from file
        event_times = parse_trace(args.trace_dir)
        # asyncio.run( replay_traces(event_times) )
        # time.sleep(180)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(replay_traces( event_times, load ))
