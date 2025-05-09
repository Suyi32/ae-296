# Instructions


First, please clone the repo and change directory (cd) to the repo.

We provide a docker container that have the environments installed.

We will provide an public IP. And you can logon the instance with the command ``ssh ubuntu@<IP>``. The password is **atc2025ae**.


- Run the container

  ``docker run --gpus all --shm-size 16G -d --name atc-ae atc-ae:latest``

- Enter the container

  ``docker exec -it atc-ae /bin/bash``

- (Optional) We have intalled Oh-my-zsh, you can use it using the following command.

  ``zsh``

- Go to the repo

  ``cd ae-296``

- Ensure the repo is up-to-date.

  ``git pull``
  
### Reproduce the CPU-assisted LoRA Computation

#### Use cached LoRA. 
- Spinup the server. The server needs some time (30~60s) to launch.

  ``python3 runServerCPULoRA.py ./config/ali-a10-cached.yml``

- Please wait 30~60s for the server to be ready. You can check the server log in ``./logs/cached_lora/launch_server.log`` to see if the server is ready. The server is ready if you see ``INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)`` in the log file.
- Open another terminal, enter the container, and change directory (cd) to the repo: ``cd ae-296``
- Go to the benchmark directory and send a small number of requests

  ``cd benchmark``
  
  ``python send_req_ae_test.py``

- In the terminal that sends requests, you will see the total time costs of serving the requests.
- Press Ctrl + C to terminate the server.

#### Load LoRA on demand. 
- Spinup the server. The server needs some time (30~60s) to launch.

  ``python3 runServerCPULoRA.py ./config/ali-a10-ondmd.yml``

- Please wait 30~60s for the server to be ready. You can check the server log in ``./logs/ondmd_lora/launch_server.log`` to see if the server is ready. The server is ready if you see ``INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)`` in the log file.
- Open another terminal, enter the container, and change directory (cd) to the repo: ``cd ae-296``
- Go to the benchmark directory and send a small number of requests

  ``cd benchmark``
  
  ``python send_req_ae_test.py``

- In the terminal that sends requests, you will see the total time costs of serving the requests.
- Press Ctrl + C to terminate the server.

#### CPU-assisted Computation. 
- Spinup the server. The server needs some time (30~60s) to launch.

  ``python3 runServerCPULoRA.py ./config/ali-a10.yml``

- Please wait 30~60s for the server to be ready. You can check the server log in ``./logs/toppings_lora/launch_server.log`` to see if the server is ready. The server is ready if you see ``INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)`` in the log file.
- Open another terminal, enter the container, and change directory (cd) to the repo: ``cd ae-296``
- Go to the benchmark directory and send a small number of requests

  ``cd benchmark``

  ``python send_req_ae_test.py``

- In the terminal that sends requests, you will see the total time costs of serving the requests.
- Press Ctrl + C to terminate the server.

#### Compare the results.
The expected results: the time costs of *CPU-assisted Computation* can rival that of *Use cached LoRA* and outperform *Load LoRA on demand*.

That is: *Use cached LoRA* < *CPU-assisted Computation* < *Load LoRA on demand*.

#### Parse the server logs.
- Go to the directory ``./ae-296``. Using the below command to see the average prefilling time, decoding time and LoRA loading time on the console.

  ``python parse_server_log.py``


### Reproduce the scheduler performance
- Enter the scheduler folder
``cd scheduler``

- We have downloaded the trace. You can also do it by yourself: Download the Azure Function Trace that will generate request traffic ([link](https://drive.google.com/file/d/1FslyHq-PIS8kutA48Ox9A4z0FNQjUZyL/view?usp=drive_link)) and place it under ``./scheduler``

- Run the experiments, which will last for minutes.
``python3 -m scheduler.run``

- The results will be present the figure. You can compare the results with those in Figure 19.
  ``schduler_results_1.pdf`` and ``schduler_results_2.pdf``