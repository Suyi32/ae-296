import argparse
import os
import psutil
import sys

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1, help="number of GPUs")
    parser.add_argument("--model_dir",  type=str, default="/root/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16", 
                            help="model path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--input_len",  type=int, default=64, help="prompt length")
    parser.add_argument("--output_len", type=int, default=10, help="max output length")
    parser.add_argument("--lora_rank",  type=int, default=64, help="lora rank")
    parser.add_argument("--lora_num",   type=int, default=1, help="number of CPU LoRAs")
    parser.add_argument("--repeat_times",   type=int, default=5, help="number of repeated runs")
    # parser.add_argument("--mode",       type=str, default="base", choices=["base", "gpu_lora", "cpu_lora", "aaas-gpu", "aaas"], help="run mode")
    args = parser.parse_args() 
    print(args)
    
    # set self.max_wait_tokens in router manager to 0
    # running_max_req_size: max batch size allowed
    cmd = "python -m lightllm.server.api_server --host 127.0.0.1 --port 8080 --model_dir {} \
             --running_max_req_size 32 --max_req_input_len 16 --tp 1 --max_total_token_num 5120 --mode aaas".format(args.model_dir)

    print("CMD in launch_server:", cmd)
    os.system(cmd)