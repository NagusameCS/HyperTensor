// scripts/paperA_proof/l2_thrash.cu
// -----------------------------------------------------------------------------
// L2 cache thrash co-tenant kernel for Experiment B.
//
// Purpose: Stream a buffer of size THRASH_MB through L2 continuously while
// the geodessical inference process runs concurrently on the same GPU.
// This reduces the effective L2 capacity available to the inference
// workload by approximately THRASH_MB.
//
// Theory of operation:
//   - The thrash kernel reads a buffer of THRASH_MB into registers in a
//     loop. Each iteration touches every cache line once, defeating LRU
//     replacement and forcing eviction of competing data.
//   - The inference process sees an effective L2 of (32 - THRASH_MB) MB.
//   - If the GRC super-baseline is L2-fit-driven, it should shrink as
//     THRASH_MB approaches 22 MB (point at which 32 - THRASH_MB drops
//     below the predicted GRC working set of ~10 MB).
//   - If the super-baseline is fusion-driven, it should be flat in
//     THRASH_MB.
//
// Caveat (Ada / RTX 4070): unlike A100, this GPU does NOT expose
// cudaLimitPersistingL2CacheSize for clean partition. The co-tenant
// approach also burns SM cycles, which is a confound for any
// throughput-based measurement; we therefore measure NCU-reported L2
// hit-rate and DRAM bytes directly, not wall-clock throughput.
//
// Build:
//   nvcc -O3 -arch=sm_89 -o l2_thrash.exe l2_thrash.cu
//
// Run (24 MB thrash, infinite loop, runs until killed):
//   l2_thrash.exe 24
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                     __FILE__, __LINE__, cudaGetErrorString(err));       \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)
#endif

// Stream-read kernel: every thread reads N elements stride-1 starting from
// blockIdx.x * blockDim.x + threadIdx.x. We use uint4 loads (16 B/thread/iter)
// to maximise memory issue bandwidth.
__global__ void thrash_kernel(const uint4* __restrict__ buf,
                              size_t n_elems_uint4,
                              uint32_t* __restrict__ sink)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (size_t i = tid; i < n_elems_uint4; i += stride) {
        uint4 v = buf[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    // Prevent the compiler from eliding the loads.
    if (tid == 0) {
        sink[0] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    }
}

int main(int argc, char** argv)
{
    int thrash_mb = (argc > 1) ? std::atoi(argv[1]) : 24;
    int run_seconds = (argc > 2) ? std::atoi(argv[2]) : 0;  // 0 = until killed
    if (thrash_mb <= 0 || thrash_mb > 1024) {
        std::fprintf(stderr, "Usage: l2_thrash <thrash_mb> [run_seconds]\n");
        return 1;
    }

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("[l2_thrash] device=%s L2=%d MB SMs=%d\n",
                prop.name, prop.l2CacheSize / (1024 * 1024),
                prop.multiProcessorCount);

    size_t bytes = (size_t)thrash_mb * 1024 * 1024;
    size_t n_uint4 = bytes / sizeof(uint4);
    std::printf("[l2_thrash] streaming %d MB (%zu uint4) continuously\n",
                thrash_mb, n_uint4);

    uint4* d_buf = nullptr;
    uint32_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_buf, 0xA5, bytes));

    int threads = 256;
    // Use enough blocks to fill the GPU but not so many that the kernel
    // takes too long per launch (we want fast iteration so the data
    // stays hot in L2 if it would otherwise persist).
    int blocks = prop.multiProcessorCount * 4;

    std::printf("[l2_thrash] launching grid=(%d,1,1) block=(%d,1,1)\n",
                blocks, threads);

    auto start = std::chrono::steady_clock::now();
    uint64_t iters = 0;
    while (true) {
        thrash_kernel<<<blocks, threads>>>(d_buf, n_uint4, d_sink);
        // No sync inside the loop; let launches queue.
        iters++;
        if ((iters & 0xFF) == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            if (run_seconds > 0 && elapsed >= run_seconds) break;
            // Also drain stdout periodically so the harness can see progress.
            if (elapsed % 10 == 0) {
                std::printf("[l2_thrash] iter=%llu elapsed=%llds\n",
                            (unsigned long long)iters, (long long)elapsed);
                std::fflush(stdout);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_sink));
    std::printf("[l2_thrash] finished after %llu iters\n",
                (unsigned long long)iters);
    return 0;
}
