import os
import subprocess
import time
import re
import yaml
import sys


def find_available_cpus():
    pattern = r"On-line CPU\(s\) list:\ *(\d+-\d+(?:,\d+-\d+)*)"
    proc = subprocess.Popen("lscpu", stdout=subprocess.PIPE)
    proc.wait()
    out = str(proc.communicate()[0])
    online_cpu_sets = re.search(pattern, out).group(1).split(",")
    available_cpus = []
    for cpu_set in online_cpu_sets:
        start, end = cpu_set.split("-")
        available_cpus += list(range(int(start), int(end) + 1))
    return available_cpus


def create_lora_loader(logFolder, world_size, hidden_dim, env_, n_layers, lora_rank, max_prefill_batch_size):
    os.makedirs(f"{logFolder}/loraLoaderLog", exist_ok=True)
    processes = []
    # for each gpu, create swapper
    for i in range(world_size):
        logpath = f"{logFolder}/loraLoaderLog/gpu_{i}.log"
        p = subprocess.Popen(
            f"exec python shared_tensor.py --gpu_rank {i} --tp_degree {world_size} --max_prefill_batch_size {max_prefill_batch_size} --rank {lora_rank} --hidden_dim {hidden_dim} --n_layers {n_layers} > {logpath} 2>&1",
            shell=True, env=env_)
        processes.append(p)
    return processes


def create_shm(logFolder, loras_num, env_, available_cpus, tp_degree, hidden_dim, max_prefill_batch_size):
    # shm uses the first (0th) available cpus
    shm_cpu = available_cpus[0]
    print("CPU of shm:", shm_cpu)
    print(f"Max prefill batch size: {max_prefill_batch_size}")
    create_shm_command = f"exec numactl --physcpubind={shm_cpu} python create_shm.py --batch_size {loras_num} --lora_num {loras_num} --tp_degree {tp_degree} --max_prefill_batch_size {max_prefill_batch_size} --hidden_dim {hidden_dim} "
    create_shm_log = os.path.join(logFolder, f"create_shm_{loras_num}.log")
    create_shm_command += f">{create_shm_log} 2>&1"
    create_shm_process = subprocess.Popen(
        create_shm_command, shell=True, env=env_)
    print("create_shm_pid PID:", create_shm_process.pid)
    return [create_shm_process]


def create_model(logFolder, world_size, input_len, loras_num, lora_rank, model_dir, mode, env_, available_cpus, max_prefill_batch_size, token_per_lora):
    """
        running_max_req_size: max decoding batch size support
        max_req_input_len: the max value for req input tokens num
    """
    # model uses 1-4th available cpus
    model_cpus = ",".join([str(i) for i in available_cpus[1:5]])
    print("CPU of model:", model_cpus)
    # test_command = f"exec numactl --physcpubind={model_cpus} python -W ignore -m lightllm.server.api_server --host 127.0.0.1 --port 8080 "
    # test_command += f"--model_dir {model_dir} --running_max_req_size 64 --max_req_input_len {input_len} --tp {world_size} --max_prefill_batch_size {max_prefill_batch_size} "
    # test_command += f"--max_total_token_num 50000 --lora_num {loras_num} --lora_rank {lora_rank} --mode {mode} "
    # test_command += f"--token_per_lora {token_per_lora} --disable_log_stats --max_req_total_len 768 "

    test_command = f"exec numactl --physcpubind={model_cpus} python -W ignore -m lightllm.server.api_server --host 127.0.0.1 --port 8080 "
    test_command += f"--model_dir {model_dir} --running_max_req_size 16 --max_req_input_len {input_len} --tp {world_size} --max_prefill_batch_size {max_prefill_batch_size} "
    test_command += f"--max_total_token_num 5000 --lora_num {loras_num} --lora_rank {lora_rank} --mode {mode} "
    test_command += f"--token_per_lora {token_per_lora} --disable_log_stats --max_req_total_len 320 "

    test_log = os.path.join(logFolder, "launch_server.log")
    test_command += f"> {test_log} 2>&1"
    test_process = subprocess.Popen(test_command, shell=True, env=env_)
    print("base PID:", test_process.pid)
    return [test_process]


def create_loras(logFolder, project_path, cpu_num, loras_num, env_, available_cpus, hidden_dim, lora_rank, token_per_lora):
    print("====== Spin up lora cntrs...")
    # lora uses the rest cpus
    cpu_set = available_cpus[5:]
    assert cpu_num <= len(cpu_set), f"Error: no enough CPUs for LoRAs ({cpu_num} > {len(cpu_set)})"
    lora_processes = []
    for i in range(cpu_num):
        print(f"CPU of lora{i}:", cpu_set[i])
        path = os.path.join(project_path, 'lora_cpu', 'lora_shm_op', 'app.py')
        lora_command  = f"exec numactl --physcpubind={cpu_set[i]} python {path} "
        lora_command += f"--batch_size {loras_num} --signal_index {i} --in_size {hidden_dim} --out_size {hidden_dim} --rank {lora_rank} "
        lora_command += f"--offset {token_per_lora} "
        lora_log = os.path.join(logFolder, "loraLogs", f"lora_{i}.log")
        lora_command += f"> {lora_log} 2>&1"
        lora_process = subprocess.Popen(lora_command, shell=True, env=env_)
        lora_processes.append(lora_process)
    return lora_processes


def cleanup(all_procs):
    print("Cleaning up...")
    # kill all
    for p in all_procs:
        print("kill", p.pid)
        # maybe we need kill using sigkill?
        os.system(f"kill -TERM {p.pid} > /dev/null 2>&1")


def main():
    assert len(sys.argv) >= 2, "python runServerCPULoRA.py <config_path>"
    available_cpus = find_available_cpus()
    print("available_cpus", available_cpus)
    project_path = os.getcwd()
    print("Project Path", project_path)

    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), "Config load failed"


    ## model spec related config
    model_dir = config['model_dir']
    hidden_dim = config.get("hidden_dim", 4096)
    n_layers = config.get("n_layers", 32)

    # model runtime config
    world_size = config.get('world_size', 1)
    max_prefill_batch_size = config.get("max_prefill_batch_size", 1)
    input_len = config.get('input_len', 64) # max input length supported

    # lora config
    token_per_lora = config.get("token_per_lora", 16)
    lora_rank = config.get("lora_rank", 64)
    loras_num = ((input_len - 1) // token_per_lora + 1) * max_prefill_batch_size
    cpu_num = loras_num * 3
    mode = config.get('mode', 'base')
    assert mode in ['base', 'aaas-gpu', 'aaas', 'aaas-cached'], f"Mode {mode} is not supported"
    all_procs = []

    shm_name_folder = os.path.join(project_path, "lora_cpu", "shm_names")
    shm_env = os.environ.copy()
    shm_env['shm_name_folder'] = shm_name_folder
    shm_env['mode'] = mode
    print("shm_name_folder", shm_name_folder)

    current_datetime = time.strftime("%m%d_%H%M%S")

    if mode == "aaas-cached":
        baseline_name = "cached_lora"
    elif mode == "aaas-gpu":
        baseline_name = "ondmd_lora"
    elif mode == "aaas" and "-token-" not in sys.argv[1]:
        baseline_name = "toppings_lora"
    elif "-token-" in sys.argv[1]:
        token_num = sys.argv[1].split("-token-")[1].split(".yml")[0]
        baseline_name = "toppings-token-" + token_num

    logFolder = os.path.join(project_path, "logs", baseline_name)

    os.makedirs(logFolder, exist_ok=True)
    os.makedirs(os.path.join(logFolder, "loraLogs"), exist_ok=True)
    os.system('cp {} {}'.format(sys.argv[1], logFolder)) # cp config file to log folder

    # Remove all files in the shm_name_folder
    file_list = os.listdir(shm_name_folder)
    for file in file_list:
        os.remove(os.path.join(shm_name_folder, file))


    try:
        create_shm_process = create_shm(logFolder, loras_num, shm_env, available_cpus, world_size, hidden_dim, max_prefill_batch_size)
        all_procs += create_shm_process
        time.sleep(10)

        loader_processes = create_lora_loader(logFolder, world_size, hidden_dim, shm_env, n_layers, lora_rank, max_prefill_batch_size)
        all_procs += loader_processes
        time.sleep(1)

        model_process = create_model(
            logFolder, world_size, input_len, loras_num, lora_rank, model_dir, mode, shm_env, available_cpus, max_prefill_batch_size, token_per_lora)
        all_procs += model_process
        time.sleep(3)

        lora_procs = create_loras(logFolder, project_path, cpu_num, loras_num, shm_env, available_cpus, hidden_dim, lora_rank, token_per_lora)
        all_procs += lora_procs

        model_process[0].wait()
    except Exception as e:
        print("Error:", e)
    finally:
        cleanup(all_procs)


if __name__ == '__main__':
    main()
