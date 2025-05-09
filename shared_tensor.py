import torch
from torch import nn

import zmq
import time
import os
import sys
import signal
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from multiprocessing import shared_memory
import threading
import tensor_cp

import numpy as np
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import json
import gc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_prefill_batch_size", type=int, default=2, help="max_prefill_batch_size")
parser.add_argument("--tp_degree", type=int, default=1, help="tensor parallel degree")
parser.add_argument("--gpu_rank", type=int, default=0, help="the rank of the gpu")
parser.add_argument("--hidden_dim", type=int, default=4096, help="hidden dimension")
parser.add_argument("--rank", type=int, default=64, help="rank of lora")
parser.add_argument("--n_layers", type=int, default=32, help="number of layers")

args = parser.parse_args()
print(args)

device = f"cuda:{args.gpu_rank}"
torch.cuda.set_device(device)
torch.set_num_threads(1)
print("Number of cpu threads: {}".format(torch.get_num_threads()))

signal_freq = 10

configJson = {}
n_layers = args.n_layers
configJson["rank"] = args.rank
configJson["hidden_dim"] = args.hidden_dim
configJson["tp_degree"] = args.tp_degree
configJson["max_prefill_batch_size"] = args.max_prefill_batch_size
configJson["n_layers"] = args.n_layers

dummy_lora_down_cpu = [ 0.001 * torch.randn( (configJson["hidden_dim"], configJson["rank"]), device="cpu", dtype=torch.float16).transpose(-1, -2).contiguous() for _ in range(n_layers) ]
dummy_lora_up_cpu = [ 0.001 * torch.randn( (configJson["rank"], configJson["hidden_dim"]//configJson["tp_degree"]), device="cpu", dtype=torch.float16).transpose(-1, -2).contiguous() for _ in range(n_layers) ]


def signal_term_handler(signal, frame):
    print("The swapper process is killed gracefully.")
    sys.exit(0)


shm_name_folder = os.getenv("shm_name_folder")
print(f"path: {shm_name_folder}")

shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "progress" in item ]
assert len(shm_name_files) == 1, "{}".format(len(shm_name_files))
shm_filename = shm_name_files[0]
shm_name = shm_filename[:-4].split("-")[1]
shm_progress = shared_memory.SharedMemory(name=shm_name)
progress = np.ndarray(args.tp_degree * args.max_prefill_batch_size, dtype=np.int8, buffer=shm_progress.buf)
progress_id = args.gpu_rank * args.max_prefill_batch_size


def main():
    logging.critical(f"gpu id: {args.gpu_rank}")
    signal.signal(signal.SIGTERM, signal_term_handler)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind('tcp://*:{}'.format( 6000 + 100 * args.gpu_rank))
    
    shared_tensor_list = []
    counter = 0
    while True:

        cuda_tensor_info = sock.recv_pyobj()
        rebuilt_tensor = rebuild_cuda_tensor(torch.Tensor, **cuda_tensor_info)
        shared_tensor_list.append(rebuilt_tensor)

        counter += 1
        logging.info("rebuilt_tensor shape: {}. Counter: {}.".format(rebuilt_tensor.shape, counter))
        sock.send_string('')

        # break 
        if counter == 6:
            break
    
    logging.info("All {} loras has been shared.".format(len(shared_tensor_list)))
    for tensor_id, item in enumerate(shared_tensor_list):
        print("Tensor {}: {}".format(tensor_id, item.shape))

    context_pull = zmq.Context()
    socket_pull = context_pull.socket(zmq.PULL)
    socket_pull.bind('tcp://*:{}'.format( 5550 + 100 * args.gpu_rank))
    #Creating poller
    poller = zmq.Poller()
    poller.register(socket_pull, zmq.POLLIN)

    #Creating a context
    context_push = zmq.Context()
    #creating and connceting to a socket.
    socket_push = context_push.socket(zmq.PUSH)
    socket_push.connect('tcp://localhost:{}'.format( 5551 + 100 * args.gpu_rank))

    torch.cuda.empty_cache()
    gc.collect()
    while True:
        start_layer_id = 8
        for j in range(args.max_prefill_batch_size):
            if progress[progress_id + j] == 1:
                swap_start = time.time()
                # for layer_id in range(n_layers):
                    # qkv up

                for k in range(start_layer_id, n_layers):
                    # mapping: 0 -> q, 1 -> k, 2 -> v
                    for i in range(1):
                        shared_tensor_list[i][j,k] = torch.empty_like(dummy_lora_up_cpu[k]).copy_(dummy_lora_up_cpu[k])

                    for i in range(1, 3):
                        shared_tensor_list[i][j,k] = torch.empty_like(dummy_lora_up_cpu[k]).copy_(dummy_lora_up_cpu[k])

                    for i in range(3, 6): 
                        shared_tensor_list[i][j,k] = torch.empty_like(dummy_lora_down_cpu[k]).copy_(dummy_lora_down_cpu[k])

                    if k == 15:
                        torch.cuda.synchronize()
                        progress[progress_id + j] = 2
                    if k == 23:
                        torch.cuda.synchronize()
                        progress[progress_id + j] = 3
                    if k == 31:
                        torch.cuda.synchronize()
                        progress[progress_id + j] = 4
                
                for k in range(0, start_layer_id):
                    for i in range(1): 
                        shared_tensor_list[i][j,k] = torch.empty_like(dummy_lora_up_cpu[k]).copy_(dummy_lora_up_cpu[k])

                    for i in range(1, 3):
                        shared_tensor_list[i][j,k] = torch.empty_like(dummy_lora_up_cpu[k]).copy_(dummy_lora_up_cpu[k])

                    for i in range(3, 6): 
                        shared_tensor_list[i][j,k] = torch.empty_like(dummy_lora_down_cpu[k]).copy_(dummy_lora_down_cpu[k])

                torch.cuda.synchronize()
                progress[progress_id + j] = 0
                torch.cuda.empty_cache()

                logging.info("End. Swap latency: {:.3f}. Progress: {}, id: {}, progress: {}".format(1000*(time.time() - swap_start), progress[progress_id], progress_id, progress))
if __name__ == "__main__":
    main()
