#pragma once
#include <iostream>
#include <functional>
#include <chrono>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


constexpr uint32_t MAX_TOK_LEN = 256;
constexpr uint32_t MAX_BATCH_SIZE = 64;
constexpr uint32_t HIDDEN_DIM = 5120;
constexpr uint32_t FP16_NBYTES = 2;

// this stream is used in collecting data from host
cudaStream_t stream;
cudaStream_t up_stream;
cudaStream_t down_stream;
cudaEvent_t event;


namespace detail {
    template<class T, T... inds, class F>
    constexpr void loop(std::integer_sequence<T, inds...>, F&& f) {
        (f(std::integral_constant<T, inds>{}), ...);
    }
}// detail

void create_stream_with_priority(int _) {
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
    cudaEventCreate(&event);
}

void create_stream(int _) {
    cudaStreamCreate(&stream);
    cudaStreamCreateWithFlags(&up_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&down_stream, cudaStreamNonBlocking);

}

void stream_synchronize(int _) {
    cudaStreamSynchronize(stream);
}

void record_event(int _) {
    cudaEventRecord(event, stream);
    cudaStreamWaitEvent(0, event);
}


template<class T, T count, class F>
constexpr void loop(F&& f) {
  detail::loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}


uint64_t get_ptr(at::Tensor& tensor) {
    auto float_ptr = tensor.data_ptr<torch::Half>();
#ifdef _DEBUG
    std::cout << "from cpp ext:" << float_ptr <<"\n"
        << "dtype " << tensor.dtype() << "\n"
        << "device " << tensor.device() << "\n"
        << "size " << tensor.size(0) << std::endl;
#endif
    return reinterpret_cast<uint64_t>(float_ptr);
}


template <int BATCH>
void to_host(at::Tensor& shm_tensor, uint64_t cuda_addr_, const uint32_t tok_num) {
    auto shm_addr = shm_tensor.data_ptr<torch::Half>();
    auto cuda_addr = reinterpret_cast<short*>(cuda_addr_);

    auto offset = shm_tensor.nbytes() / MAX_BATCH_SIZE / FP16_NBYTES;
    auto sliced_bytes = sizeof(short) * HIDDEN_DIM * tok_num;
    loop<int, BATCH>(
        [=](auto i) {
            cudaMemcpy(
                shm_addr + i * offset,
                cuda_addr + i * offset,
                sliced_bytes, cudaMemcpyDeviceToHost
            );
        }
    );
}


template <int BATCH>
void to_device(at::Tensor& shm_tensor, uint64_t cuda_addr_, const uint32_t tok_num) {
    auto shm_addr = shm_tensor.data_ptr<torch::Half>();
    auto cuda_addr = reinterpret_cast<short*>(cuda_addr_);

    auto offset = shm_tensor.nbytes() / MAX_BATCH_SIZE / FP16_NBYTES;
    auto sliced_bytes = sizeof(short) * HIDDEN_DIM * tok_num;
    loop<int, BATCH>(
        [=](auto i) {
            cudaMemcpy(
                cuda_addr + i * offset,
                shm_addr + i * offset,
                sliced_bytes, cudaMemcpyHostToDevice
            );
        }
    );
}


void register_pinned_memory(uint64_t tensor, size_t nbytes) {
    void* addr = reinterpret_cast<void*>(tensor);
    cudaHostRegister(addr, nbytes, cudaHostAllocPortable);
}


void cudaDtoH(at::Tensor& src_, uint64_t dst_, const size_t offset, const size_t nbytes) {
    auto src = src_.data_ptr<torch::Half>();
    auto dst = reinterpret_cast<short*>(dst_);
    cudaMemcpy(dst, src + offset, nbytes, cudaMemcpyDeviceToHost);
}


void cudaDtoHAsync(at::Tensor& src_, uint64_t dst_, const size_t offset, const size_t nbytes) {
    auto src = src_.data_ptr<torch::Half>();
    auto dst = reinterpret_cast<short*>(dst_);
    cudaMemcpyAsync(dst, src + offset, nbytes, cudaMemcpyDeviceToHost);
}

void cudaSignalDtoHAsync(at::Tensor& src_, uint64_t dst_, const size_t offset, const size_t nbytes) {
    auto src = src_.data_ptr<short>();
    auto dst = reinterpret_cast<short*>(dst_);
    cudaMemcpyAsync(dst, src + offset, nbytes, cudaMemcpyDeviceToHost);
}

void cudaDtoHAsyncWithSignal(at::Tensor& src_, uint64_t dst_, const size_t offset, const size_t nbytes, uint64_t sig_src_, size_t index, uint64_t sig_dst_) {

    // cudaStream_t stream;
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto src = src_.data_ptr<torch::Half>();
    auto dst = reinterpret_cast<short*>(dst_);
    cudaMemcpyAsync(dst, src + offset, nbytes, cudaMemcpyDeviceToHost);
    // cudaMemcpyAsync(dst, src + offset, nbytes, cudaMemcpyDeviceToHost, stream);

    char* sig_src = reinterpret_cast<char*>(sig_src_);
    char* sig_dst = reinterpret_cast<char*>(sig_dst_);

    cudaMemcpyAsync(sig_dst, sig_src + index, 1, cudaMemcpyDeviceToHost);
    // cudaMemcpyAsync(sig_dst, sig_src + index, 1, cudaMemcpyDeviceToHost, stream);
}


void cudaHtoD(uint64_t src_, at::Tensor& dst_, const size_t offset, const size_t nbytes) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);

    cudaMemcpy(dst + offset, src, nbytes, cudaMemcpyHostToDevice);
}


void cudaHtoDAsync(uint64_t src_, at::Tensor& dst_, const size_t offset, const size_t nbytes) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);

    cudaMemcpyAsync(
                dst + offset,
                src,
                nbytes,
                cudaMemcpyHostToDevice,
                stream
            );
}


void cudaHtoD2D(uint64_t src_, at::Tensor& dst_, const size_t src_width, const size_t dst_width, const size_t width, const size_t height, const size_t offset) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);

    // cudaMemcpy2DAsync(
    //             dst + offset,
    //             dst_width,
    //             src,
    //             src_width,
    //             width,
    //             height,
    //             cudaMemcpyHostToDevice,
    //             stream
    //         );

    cudaMemcpy2D(
                dst + offset,
                dst_width,
                src,
                src_width,
                width,
                height,
                cudaMemcpyHostToDevice
            );
}

void cudaHtoD2DAsync(uint64_t src_, at::Tensor& dst_, const size_t src_width, const size_t dst_width, const size_t width, const size_t height, const size_t offset) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);

    cudaMemcpy2DAsync(
                dst + offset,
                dst_width,
                src,
                src_width,
                width,
                height,
                cudaMemcpyHostToDevice,
                stream
            );
}

void cudaHtoD2DAsyncLora(uint64_t src_, at::Tensor& dst_, const size_t src_width, const size_t dst_width, const size_t width, const size_t height, const size_t offset, int is_up) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);

    cudaMemcpy2DAsync(
                dst + offset,
                dst_width,
                src,
                src_width,
                width,
                height,
                cudaMemcpyHostToDevice,
                (is_up == 1 ? up_stream : down_stream)
            );
}