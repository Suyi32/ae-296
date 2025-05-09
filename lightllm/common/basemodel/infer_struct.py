import torch

class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.lora_ids = None
        self.b_req_idx = None
        self.b_start_loc = None
        self.b_seq_len = None
        self.max_len_in_batch = None
        self.is_prefill = None
        
        self.mem_manager = None
        self.req_manager = None
        
        self.prefill_mem_index = None
        self.prefill_key_buffer = None
        self.prefill_value_buffer = None
        
        self.decode_is_contiguous = None
        self.decode_mem_start = None 
        self.decode_mem_end = None
        self.decode_mem_index = None
        self.decode_key_buffer = None 
        self.decode_value_buffer = None

        self.aaas_mode = None
    
    def init_some_extra_state(self, 
            model, 
            batch_size, 
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            req_to_token_indexes: torch.Tensor,
            b_req_idx : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill):
        pass
