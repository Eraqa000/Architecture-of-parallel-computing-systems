%%writefile vector_add.cu

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

__global__ void Kernel(const float* A, const float* B, float* C, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t N = 100000;

    float *h_a, *h_b, *h_c, *hd_c;
    h_a  = (float*)malloc(N * sizeof(float));
    h_b  = (float*)malloc(N * sizeof(float));
    h_c  = (float*)malloc(N * sizeof(float));
    hd_c = (float*)malloc(N * sizeof(float));

    srand(123);
    for (size_t i = 0; i < N; i++) {
        h_a[i] = float(rand()) / RAND_MAX * 10.0f;
        h_b[i] = float(rand()) / RAND_MAX * 10.0f;
    }

    clock_t cpu_start = clock();

    for (size_t i = 0; i < N; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

    clock_t cpu_end = clock();
    double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    printf("CPU results:\n");
    // for (size_t i = 0; i < N; i++) {
    //    printf("i = %2zu:  %6.2f + %6.2f = %6.2f\n", i, h_a[i], h_b[i], h_c[i]);
    // }
    printf("CPU time: %.6f ms\n\n", cpu_time_ms);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 1000;
    int blocks  = (int)((N + threads - 1) / threads);

    // Таймеры CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    Kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    Kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(hd_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU results:\n");
    // for (size_t i = 0; i < N; i++) {
    //     printf("i = %2zu:  %6.2f + %6.2f = %6.2f\n", i, h_a[i], h_b[i], hd_c[i]);
    // }
    printf("GPU time: %.6f ms\n\n", gpu_time_ms);





    return 0;
}
