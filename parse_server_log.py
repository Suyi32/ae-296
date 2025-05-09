import numpy as np

server_log_folders = {
    "cached_lora": "./logs/cached_lora/launch_server.log",
    "ondmd_lora": "./logs/ondmd_lora/launch_server.log",
    "toppings_lora": "./logs/toppings_lora/launch_server.log",
}


for baseline, log_file in server_log_folders.items():
    prefill_times = []
    decode_times = []
    loading_times = []

    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "CRITICAL:[STEP] Prefill" in line:
                prefill_times.append(float(line.split(" ")[-1]))
            if "CRITICAL:[STEP] Decode" in line:
                decode_times.append(float(line.split(" ")[-1]))
            if "CRITICAL:[STEP] Prepare GPU LoRA" in line:
                loading_times.append(float(line.strip().strip(".").split(" ")[-1]))

    print(baseline)
    print("Mean prefill time: ", np.mean(prefill_times[3:]))
    print("Mean decode time: ", np.mean(decode_times[3:]))
    print("Mean loading time: ", np.mean(loading_times[3:]))
    print("--------------------------------")