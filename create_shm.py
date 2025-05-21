import signal
import time
import numpy as np
from multiprocessing import shared_memory
import os
import sys
import torch
import tensor_cp

# Gracefully shutdown
def signal_term_handler(signal, frame):
    shm_array.close()
    shm_tkn_len.close()
    shm_array.unlink()
    shm_tkn_len.unlink()
    shm_signal.close()
    shm_signal.unlink()

    # os.remove( os.path.join(shm_name_folder, shm_filename) )
    print("The shm process is killed gracefully.")
    sys.exit(0)
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--lora_num", type=int, default=1, help="number of lora launched")
parser.add_argument("--hidden_dim", type=int, default=4096, help="lora input dimension")
parser.add_argument("--prompt_max_length", type=int, default=256, help="max length of a prompt")
parser.add_argument("--tp_degree", type=int, default=1, help="tensor parallel size")
parser.add_argument("--max_prefill_batch_size", type=int, default=2, help="max prefill batch size")
args = parser.parse_args() 
print(args)

PROMPT_MAX_LENGTH = args.prompt_max_length
HIDDEN_DIM = args.hidden_dim
batch_size = args.batch_size
lora_num = args.lora_num
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_term_handler)
    try:
        shape = (batch_size, PROMPT_MAX_LENGTH, HIDDEN_DIM)
        # create shared array
        example_array = np.zeros(shape=shape, dtype=np.half)  # Start with an existing NumPy array
        shm_array = shared_memory.SharedMemory(create=True, size=2 * batch_size * PROMPT_MAX_LENGTH * HIDDEN_DIM)
        shm_out_q = shared_memory.SharedMemory(create=True, size=2 * batch_size * PROMPT_MAX_LENGTH * HIDDEN_DIM)
        shm_out_k = shared_memory.SharedMemory(create=True, size=2 * batch_size * PROMPT_MAX_LENGTH * HIDDEN_DIM)
        shm_out_v = shared_memory.SharedMemory(create=True, size=2 * batch_size * PROMPT_MAX_LENGTH * HIDDEN_DIM)
        shm_out_o = shared_memory.SharedMemory(create=True, size=2 * batch_size * PROMPT_MAX_LENGTH * HIDDEN_DIM)
        shared_array = np.ndarray(shape, dtype=np.half, buffer=shm_array.buf)
        shared_out_q = np.ndarray(shape, dtype=np.half, buffer=shm_out_q.buf)
        shared_out_k = np.ndarray(shape, dtype=np.half, buffer=shm_out_k.buf)
        shared_out_v = np.ndarray(shape, dtype=np.half, buffer=shm_out_v.buf)
        shared_out_o = np.ndarray(shape, dtype=np.half, buffer=shm_out_o.buf)

        # pin the memory to allow asynchronous writing, to do so, write it to a torch object first
        torch_shared_array = torch.from_numpy(shared_array)
        torch_shared_out_q = torch.from_numpy(shared_out_q)
        torch_shared_out_k = torch.from_numpy(shared_out_k)
        torch_shared_out_v = torch.from_numpy(shared_out_v)
        torch_shared_out_o = torch.from_numpy(shared_out_o)
        tensor_cp.register_pinned_memory(torch_shared_array.untyped_storage().data_ptr(), shared_array.nbytes)
        tensor_cp.register_pinned_memory(torch_shared_out_q.untyped_storage().data_ptr(), shared_out_q.nbytes)
        tensor_cp.register_pinned_memory(torch_shared_out_k.untyped_storage().data_ptr(), shared_out_k.nbytes)
        tensor_cp.register_pinned_memory(torch_shared_out_v.untyped_storage().data_ptr(), shared_out_v.nbytes)
        tensor_cp.register_pinned_memory(torch_shared_out_o.untyped_storage().data_ptr(), shared_out_o.nbytes)

        # create shared bytes for token length
        shm_tkn_len = shared_memory.SharedMemory(create=True, size=8)
        shared_tkn_len = shm_tkn_len.buf
        shared_tkn_len[0] = 1
        
        
        lock = shared_memory.SharedMemory(create=True, size=1)
        lock_buf = lock.buf
        lock_buf[0] = 0

        # create shared signals; Here, the number of signals == batch_size
        # store all the signal into one signal_array. In this case, we can do memcpy more efficiently.
        # 3 calls to memcpy is better than batch_size number of memcpy call that each only do very little work.
        shm_signal = shared_memory.SharedMemory(create=True, size=2*lora_num * 3)
        signal_array_np = np.ndarray((3, lora_num), dtype=np.int16, buffer=shm_signal.buf)
        torch_signal = torch.from_numpy(shared_array)
        tensor_cp.register_pinned_memory(torch_signal.untyped_storage().data_ptr(), signal_array_np.nbytes)
        torch_signal[:] = 1


        # create shared signal for checking lora weight loading progress. Can swap max_prefill_batch_size at max

        shm_progress = shared_memory.SharedMemory(create=True, size=args.tp_degree * args.max_prefill_batch_size)
        signal_progress_np = np.ndarray(args.tp_degree * args.max_prefill_batch_size, dtype=np.int8, buffer=shm_signal.buf)
        torch_progress_signal = torch.from_numpy(signal_progress_np)
        torch_progress_signal[:] = 0

        # write the shm array name
        shm_name_folder = os.getenv("shm_name_folder")

        shm_filename = "shm_array-{}.shm".format( shm_array.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_array", shm_array.name))
        shm_filename = "shm_out_q-{}.shm".format( shm_out_q.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_out_q", shm_out_q.name))
        shm_filename = "shm_out_k-{}.shm".format( shm_out_k.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_out_k", shm_out_k.name))
        shm_filename = "shm_out_v-{}.shm".format( shm_out_v.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_out_v", shm_out_v.name))
        shm_filename = "shm_out_o-{}.shm".format( shm_out_o.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_out_o", shm_out_o.name))
        # write the shm signal name
        shm_filename = "shm_signal-{}.shm".format(shm_signal.name)
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_signal", shm_signal.name))
        # write the shm token length
        shm_filename = "shm_tkn_len-{}.shm".format( shm_tkn_len.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("shm_tkn_len", shm_tkn_len.name))

        shm_filename = "shm_lock-{}.shm".format( lock.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("lock", lock.name))

        shm_filename = "shm_progress-{}.shm".format( shm_progress.name )
        with open( os.path.join(shm_name_folder, shm_filename), 'w' ) as fw:
            fw.write("{},{}\n".format("progress", shm_progress.name))
        
        print("Complete shm creation.")
        time.sleep(1800) # need to extend to longer time period. 30 min for now
    except KeyboardInterrupt:
        try:
            shm_array.close()
            shm_array.unlink()
            shm_tkn_len.close()
            shm_tkn_len.unlink()
            shm_signal.close()
            shm_signal.unlink()
            shm_progress.close()
            shm_progress.unlink()
            shm_out_q.close()
            shm_out_q.unlink()
            shm_out_k.close()
            shm_out_k.unlink()
            shm_out_v.close()
            shm_out_v.unlink()
            shm_out_o.close()
            shm_out_o.unlink()
        except FileNotFoundError:
            pass
