import signal
import sys
import argparse
import torch
import time
import os
import logging
import numpy as np
from multiprocessing import shared_memory
from functools import partial


# Gracefully shutdown
def signal_term_handler(app, signal, frame):
    for shm in app.all_shms.values():
        shm.close()
        shm.unlink()
    # app.shm_signal.close()
    # app.shm_signal.unlink()
    # os.remove( os.path.join(shm_name_folder, shm_filename) )
    print("The app process is killed gracefully.")
    sys.exit(0)


class LoRAapp:
    def __init__(self, args, shm_name_folder):
        # basic configs
        self.in_size = args.in_size
        self.out_size = args.out_size
        self.batch_size = args.batch_size
        self.signal_name = args.signal_name
        self.rank = args.rank
        self.prompt_max_length = args.prompt_max_length
        self.signal_index = args.signal_index
        self.shm_name_folder = shm_name_folder
        self.offset = args.offset
        # create the lora
        self.q_down_LoRA = torch.randn( (self.in_size, self.rank) ) * 0.001
        self.q_up_LoRA = torch.randn( (self.rank, self.out_size) ) * 0.001
        self.k_down_LoRA = torch.randn( (self.in_size, self.rank) ) * 0.001
        self.k_up_LoRA = torch.randn( (self.rank, self.out_size) ) * 0.001
        self.v_down_LoRA = torch.randn( (self.in_size, self.rank) ) * 0.001
        self.v_up_LoRA = torch.randn( (self.rank, self.out_size) ) * 0.001
        self.o_down_LoRA = torch.randn( (self.in_size, self.rank) ) * 0.001
        self.o_up_LoRA = torch.randn( (self.rank, self.out_size) ) * 0.001
        self.warm_up()
        # init shared memory
        self.init_shm()

    def warm_up(self):
        # warm up
        input_tensor = torch.rand(size=(1, 1, self.in_size))
        start_time = time.time()
        _ = input_tensor@self.q_down_LoRA@self.q_up_LoRA
        _ = input_tensor@self.k_down_LoRA@self.k_up_LoRA
        _ = input_tensor@self.v_down_LoRA@self.v_up_LoRA
        _ = input_tensor@self.o_down_LoRA@self.o_up_LoRA
        logging.info("Traced time 1: {:.6f}".format(time.time() - start_time))
        start_time = time.time()
        _ = input_tensor@self.q_down_LoRA@self.q_up_LoRA
        _ = input_tensor@self.k_down_LoRA@self.k_up_LoRA
        _ = input_tensor@self.v_down_LoRA@self.v_up_LoRA
        _ = input_tensor@self.o_down_LoRA@self.o_up_LoRA
        logging.info("Traced time 2: {:.6f}".format(time.time() - start_time))

    def init_shm(self):
        logging.info("Input shm signal name: {}.".format(self.signal_name))
        def init_shm_helper(shm_names):
            shms = {}
            for name in shm_names:
                shm_name_files = [ item for item in os.listdir(self.shm_name_folder) if ".shm" in item and name in item ]
                assert len(shm_name_files) == 1
                shm_array_filename = shm_name_files[0]
                shm_array_name = shm_array_filename[:-4].split("-")[1]
                shm_array = shared_memory.SharedMemory(name=shm_array_name)
                shms[name] = shm_array
            return shms

        all_shms = init_shm_helper(['shm_array', 'shm_out_q', 'shm_out_k', 'shm_out_v', 'shm_out_o', 'shm_tkn_len', "shm_signal"])
        self.all_shms = all_shms
        shared_array = np.ndarray((args.batch_size, args.prompt_max_length, self.in_size), np.half, buffer=self.all_shms['shm_array'].buf)
        self.torch_input_tensor = torch.from_numpy(shared_array)
        shared_out_q = np.ndarray((args.batch_size, args.prompt_max_length, self.in_size), np.half, buffer=self.all_shms['shm_out_q'].buf)
        self.torch_out_q = torch.from_numpy( shared_out_q )
        shared_out_k = np.ndarray((args.batch_size, args.prompt_max_length, self.in_size), np.half, buffer=self.all_shms['shm_out_k'].buf)
        self.torch_out_k = torch.from_numpy( shared_out_k )
        shared_out_v = np.ndarray((args.batch_size, args.prompt_max_length, self.in_size), np.half, buffer=self.all_shms['shm_out_v'].buf)
        self.torch_out_v = torch.from_numpy( shared_out_v )
        shared_out_o = np.ndarray((args.batch_size, args.prompt_max_length, self.in_size), np.half, buffer=self.all_shms['shm_out_o'].buf)
        self.torch_out_o = torch.from_numpy( shared_out_o )
        
        # signal, only care about one location in the entire signal list.
        self.sig = np.ndarray((self.signal_index+1), np.int16, buffer=(self.all_shms['shm_signal'].buf))

        # tkn length
        self.shm_tkn_len = self.all_shms['shm_tkn_len']
        self.tkn_len = self.shm_tkn_len.buf

        logging.info("Attached to shm array: {}, signal: {}, tkn_len: {}".format(self.all_shms['shm_array'].name, 
                     self.all_shms['shm_signal'].name, self.shm_tkn_len.name))

    def run(self):
        print("################# start printing ##########3\n\n\n\n\n")
        # begin computation
        with torch.no_grad():
            try:
                while True:
                    if self.sig[self.signal_index] > 1:
                        recv_time = time.time()
                        # check whether it need to do q, k, or v
                        if self.sig[self.signal_index] < 202:
                            # do q
                            row_id = self.sig[self.signal_index] - 2
                            out_tensor = self.torch_out_q
                            up_lora = self.q_up_LoRA
                            down_lora = self.q_down_LoRA
                        elif self.sig[self.signal_index] < 402:
                            # do k
                            row_id = self.sig[self.signal_index] - 202
                            out_tensor = self.torch_out_k
                            up_lora = self.k_up_LoRA
                            down_lora = self.k_down_LoRA
                        else:
                            # do v
                            row_id = self.sig[self.signal_index] - 402
                            out_tensor = self.torch_out_v
                            up_lora = self.v_up_LoRA
                            down_lora = self.v_down_LoRA

                        tkn_num = self.tkn_len[0]

                        computation_start = time.time()
                        dim_0 = row_id // ((tkn_num - 1) // self.offset + 1) # batch dimension
                        dim_1 = row_id %  ((tkn_num - 1) // self.offset + 1) # token dimension
                        dim_1_end = min( (dim_1+1)*self.offset, tkn_num )
                        input_tensor = self.torch_input_tensor[dim_0, dim_1*self.offset:dim_1_end, :self.in_size].float()
                        res = input_tensor@down_lora@up_lora
                        computation_time = time.time() - computation_start

                        write_start = time.time()
                        out_tensor[row_id, :tkn_num, :self.in_size] = res[:].half() # for tp

                        self.sig[self.signal_index] = 0
                        write_shm_time = time.time() - write_start

                        # notify the base model
                        end_time = time.time() - recv_time
                        logging.info("Row ID: {}, Server time: {:.8f}; Computation time: {:.8f}; Write time: {:.8f}".format(
                            row_id,
                            end_time,
                            computation_time,
                            write_shm_time
                            )
                        )
                        # if tkn_num > 1:
                        #     time.sleep( 0.007 ) # sleep 7 ms for prefill
                        # time.sleep( max(0.0008 - end_time,0) ) # sleep after a lora computation
                    # sleep 0.02 ms to avoid busy waiting    
                    # time.sleep(0.00001) 
            except KeyboardInterrupt:
                try:
                    for shm in self.all_shms.values():
                        shm.close()
                    for shm in self.all_shms.values():
                        shm.unlink()
                except FileNotFoundError:
                    pass



if __name__ == '__main__':
    torch.manual_seed(0)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_name", type=str, default=None, help="require a shm signal name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--in_size", type=int, default=4096, help="lora input dimension")
    parser.add_argument("--out_size", type=int, default=4096, help="lora output dimension")
    parser.add_argument("--rank", type=int, default=2, help="lora rank")
    parser.add_argument("--offset", type=int, default=16, help="offset")
    parser.add_argument("--prompt_max_length", type=int, default=256, help="max length of a prompt")
    parser.add_argument("--signal_index", type=int, default=0, help="the index in the shm_signal_array")
    
    args = parser.parse_args() 
    logging.info(args)
    # assert args.signal_name != None
    lora_app = LoRAapp(args, os.getenv("shm_name_folder"))
    partial_handler = partial(signal_term_handler, app=lora_app)
    signal.signal(signal.SIGTERM, partial_handler)
    lora_app.run()
