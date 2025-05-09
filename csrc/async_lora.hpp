#pragma once
#include <iostream>
#include <vector>
#include <filesystem>
#include <thread>
#include <atomic>
#include <future>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include "cuda_ops.hpp"
#include "tpool.hpp"
using namespace boost::interprocess;

std::vector<int*> g_shm_signal_ptrs;
int *g_tkn_num_ptr, *g_lock_buf_ptr;
std::atomic<at::Tensor*> g_res_ptr = nullptr;
std::vector<shared_memory_object*> g_shm_objs;
std::vector<mapped_region*> g_regions;
ThreadPool* g_tpool = nullptr;
std::future<void> g_res_future;
constexpr auto PROMPT_MAX_LENGTH = 256;


std::vector<std::string> get_signal_shm_names(std::string key) {
    auto shm_files_dir = std::string(std::getenv("shm_name_folder"));
    std::vector<std::string> shm_names;
    for (const auto& path : std::filesystem::directory_iterator(shm_files_dir)) {
        auto shm_name = path.path().filename().string();
        if (shm_name.substr(4, key.length()) != key) continue;
        size_t begin = shm_name.find_first_of('-');
        shm_name = shm_name.substr(begin + 1, shm_name.length() - begin - 5);
        shm_names.push_back(shm_name);
        std::cout << shm_name << '\n';
    }
    return shm_names;
}


std::vector<int*> get_signal_ptrs(std::vector<std::string> shm_names) {
    std::vector<int*> addrs;
    for (auto&& name: shm_names) {
        shared_memory_object* shm_obj = new shared_memory_object(open_only, name.c_str(), read_write);
        mapped_region* region = new mapped_region(*shm_obj, read_write, 0, sizeof(int));
        auto addr = static_cast<int*>(region->get_address());
        // register pinned memory
        cudaHostRegister(addr, sizeof(uint8_t), cudaHostAllocPortable);
        addrs.push_back(addr);
        g_shm_objs.push_back(shm_obj);
        g_regions.push_back(region);
    }
    return addrs;
}


int register_shm_signals(int unused){
    g_shm_signal_ptrs = get_signal_ptrs(get_signal_shm_names("signal"));
    g_tkn_num_ptr = get_signal_ptrs(get_signal_shm_names("tkn_len"))[0];
    g_lock_buf_ptr = get_signal_ptrs(get_signal_shm_names("lock"))[0];
    g_tpool = new ThreadPool(1);
    return 0;
}


void cpu_lora_thread(at::Tensor hidden_states, int tkn_num, std::vector<int> lora_inds,
                     int batch_size, int hidden_dim, int tp_degree, uint64_t torch_res_ptr,
                     uint64_t number_ptr, int gpu_id, int layer_id, int n_layers)
{
    // set sig to -1.
    if (gpu_id == 0) {
        for (auto shm_id : lora_inds) {
            g_shm_signal_ptrs[shm_id][0] = 1;
        }
        g_lock_buf_ptr[0] = (layer_id) % n_layers + 1;
    }
    else {
        while (g_lock_buf_ptr[0] != (layer_id + 1));
    }
    // set tkn length
    g_tkn_num_ptr[0] = tkn_num;
    // begin copy to cpu
    for (size_t row_id = 0; row_id <lora_inds.size(); row_id++){
        int shm_id = lora_inds[row_id];
        cudaDtoHAsyncWithSignal(hidden_states,
            torch_res_ptr + 2 * hidden_dim * PROMPT_MAX_LENGTH * row_id,
            hidden_dim * tkn_num * shm_id,
            tkn_num * hidden_dim,
            number_ptr,
            row_id + 2,
            reinterpret_cast<uint64_t>(g_shm_signal_ptrs[shm_id]));
    }

    auto q_lora_res = torch::zeros(
        {batch_size, tkn_num, hidden_dim/tp_degree},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, gpu_id).requires_grad(false));

    int checker = (1 << lora_inds.size()) - 1;
    while (checker > 0) {

        for (size_t i = 0; i < lora_inds.size(); i++) {
            auto e = lora_inds[i];
            if ((checker & (1 << i)) && g_shm_signal_ptrs[i][0] == 0) {
                for (size_t j = 0; j < tkn_num; j++) {
                    cudaHtoDAsync(
                        torch_res_ptr + 2 * ((hidden_dim/tp_degree) * gpu_id + i * PROMPT_MAX_LENGTH * hidden_dim + j * hidden_dim), 
                        q_lora_res,
                        (hidden_dim/tp_degree) * gpu_id + e * tkn_num * hidden_dim + j * hidden_dim, 
                        hidden_dim/tp_degree
                    );
                }
                checker -= (1 << i);
            }
        }
    }
    g_res_ptr = new at::Tensor(std::move(q_lora_res));
}


int invoke_cpu_lora(at::Tensor hidden_states, int tkn_num, std::vector<int> lora_inds,
                     int batch_size, int hidden_dim, int tp_degree, uint64_t torch_res_ptr,
                     uint64_t number_ptr, int gpu_id, int layer_id, int n_layers)
{
    // clear old results;
    if (g_res_ptr) delete g_res_ptr;
    g_res_ptr = nullptr;
    g_res_future = g_tpool->enqueue(cpu_lora_thread, hidden_states, tkn_num, lora_inds, batch_size,
                   hidden_dim, tp_degree, torch_res_ptr, number_ptr, gpu_id, layer_id, n_layers);
    return 0;
}


at::Tensor collect(int tid) {
    g_res_future.wait();
    if (!g_res_ptr) {
        std::cerr << "Error: Receives an nullptr in collect phase" << std::endl;
        // just return a non-null value to avoid segment fault
        return torch::zeros(5);
    }
    return (*g_res_ptr);
}


int dealloc_resource() {
    if (g_res_ptr) delete g_res_ptr;
    for (auto&& shm_obj_ptr: g_shm_objs) delete shm_obj_ptr;
    for (auto&& region_ptr: g_regions) delete region_ptr;
    delete g_tpool;
    return 0;
}
