from torch import nn
import time
import torch

# Lora Class
print("[Using customized modeling_llama]")
class LoRA(nn.Module):
    def __init__(self, in_size=4096, out_size=4096, rank=2):
        super(LoRA, self).__init__()
        self.fc1 = nn.Linear(in_features=in_size, out_features=rank, bias=False)
        self.fc2 = nn.Linear(in_features=rank, out_features=out_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # create the lora
    hidden_dim = 4096
    lora_function = LoRA(rank=64).to( torch.device('cpu') )
    input_tensor = torch.rand(size=(1,hidden_dim))
    input_tensor_1 = torch.rand(size=(1,hidden_dim))
    input_tensor_2 = torch.rand(size=(1,hidden_dim))
    input_tensor_3 = torch.rand(size=(1,hidden_dim))
    traced_lora = torch.jit.trace(lora_function, (input_tensor))

    # warm up
    start_time = time.time()
    with torch.no_grad():
        output = traced_lora(input_tensor)
    print("Traced time 1: {:.6f}".format(time.time() - start_time))

    start_time = time.time()
    with torch.no_grad():
        output = traced_lora(input_tensor)
    print("Traced time 2: {:.6f}".format(time.time() - start_time), output.shape)

    a = torch.rand((4096,2))
    b = torch.rand((2,4096)) 
    # input_tensor = torch.rand(size=(1,1,hidden_dim), dtype=torch.float16)
    start_time = time.time()
    # c = torch.bmm(torch.bmm(input_tensor, a,), b)
    c = input_tensor@a@b
    print("Traced time 3: {:.6f}".format(time.time() - start_time))
    print(c.shape)

    a = torch.rand((4096,4096))
    # input_tensor = torch.rand(size=(1,1,hidden_dim), dtype=torch.float16)
    start_time = time.time()
    # c = torch.bmm(torch.bmm(input_tensor, a,), b)
    c = input_tensor@a
    print("Traced time 4: {:.6f}".format(time.time() - start_time))
    print(c.shape)
