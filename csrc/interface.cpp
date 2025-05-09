/**
 * @file interface.cpp
 * @brief fast data movement between GPU and CPU memory
 * @author madoka
 */
#include <iostream>
#include <functional>
#include <chrono>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_ops.hpp"
// #include "async_lora.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "fast data movement between GPU and CPU memory";

    // get data pointer of a CUDA tensor
    m.def("get_ptr", &get_ptr, "get ptr of cuda tensor", py::arg("tensor"));

    // create a stream for data transfer
    m.def("create_stream", &create_stream, "create a stream for data transfer", py::arg("_"));

    // create a stream for data transfer
    m.def("stream_synchronize", &stream_synchronize, "synchronize stream", py::arg("_"));

    // record a cuda event
    m.def("record_event", &record_event, "record a cuda event", py::arg("_"));

    // copy CUDA tensor to CPU
    m.def("to_host", &to_host<8>, "copy from device to host",
        py::arg("cpu_tensor"),  py::arg("cu_addr"), py::arg("tok_num"));

    // copy CPU tensor to CUDA device
    m.def("to_device", &to_device<8>, "copy from host to device",
        py::arg("cpu_tensor"),  py::arg("cu_addr"), py::arg("tok_num"));

    // register pinned host memory
    m.def("register_pinned_memory", &register_pinned_memory,
        "register pinned memory", py::arg("tensor"), py::arg("nbytes"));

    // move data from device to host
    m.def("cudaDtoH", &cudaDtoH,
        "move data from device to host", py::arg("src"),  py::arg("dst"), py::arg("offset"), py::arg("nbytes"));

    // move data from device to host
    m.def("cudaDtoHAsync", &cudaDtoHAsync,
        "move data from device to host async", py::arg("src"),  py::arg("dst"), py::arg("offset"), py::arg("nbytes"));

    m.def("cudaSignalDtoHAsync", &cudaSignalDtoHAsync,
        "move data from device to host async", py::arg("src"),  py::arg("dst"), py::arg("offset"), py::arg("nbytes"));

    // move data from device to host
    m.def("cudaDtoHAsyncWithSignal", &cudaDtoHAsyncWithSignal,
        "move data from device to host async and send a signal", py::arg("src"),  py::arg("dst"), py::arg("offset"), py::arg("nbytes"), py::arg("sig_src"),  py::arg("sig_offset"), py::arg("sig_dst"));

    // move data from device to host
    m.def("cudaHtoD", &cudaHtoD,
        "move data from device to host", py::arg("src"),  py::arg("dst"), py::arg("offset"), py::arg("nbytes"));

    // move data from device to host
    m.def("cudaHtoDAsync", &cudaHtoDAsync,
        "move data from device to host async", py::arg("src"),  py::arg("dst"), py::arg("offset"), py::arg("nbytes"));

    // move data from device to host
    m.def("cudaHtoD2D", &cudaHtoD2D,
        "move data from device to host async", py::arg("src"),  py::arg("dst"), py::arg("src_width"), py::arg("dst_width"), py::arg("width"), py::arg("height"), py::arg("offset"));


    // move data from device to host
    m.def("cudaHtoD2DAsync", &cudaHtoD2DAsync,
        "move data from device to host async", py::arg("src"),  py::arg("dst"), py::arg("src_width"), py::arg("dst_width"), py::arg("width"), py::arg("height"), py::arg("offset"));


    m.def("cudaHtoD2DAsyncLora", &cudaHtoD2DAsyncLora,
        "move data from device to host async", py::arg("src"),  py::arg("dst"), py::arg("src_width"), py::arg("dst_width"), py::arg("width"), py::arg("height"), py::arg("offset"), py::arg("is_up"));
    // m.def("invoke_cpu_lora", &invoke_cpu_lora, "invoke cpu lora async",
    //     py::arg("hidden_states"), py::arg("tkn_num"), py::arg("lora_inds"),
    //     py::arg("batch_size"), py::arg("hidden_dim"), py::arg("tp_degree"),
    //     py::arg("torch_res_ptr"), py::arg("number_ptr"), py::arg("gpu_id"),
    //     py::arg("layer_id"), py::arg("n_layers"));
    
    // m.def("collect", &collect, "collect lora result", py::arg("tid"));
    // m.def("register_shm_signals", &register_shm_signals, "register signal shms", py::arg("unused"));
    // m.def("dealloc_resource", &dealloc_resource, "deallocate resource");
}

