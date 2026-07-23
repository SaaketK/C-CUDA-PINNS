#include "pinn/core/backend.h"

#include <stdio.h>

#ifdef PINN_USE_CUDA
    #include <cuda_runtime.h>
#endif

static int g_cuda_available = 0;
static Device g_default_device = DEVICE_CPU;
static int g_backend_initialized = 0;

int backend_init(void){
#ifdef PINN_USE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){
        g_cuda_available = 0;
        g_default_device = DEVICE_CPU;
        g_backend_initialized = 1;
        return 1;
    }

    g_cuda_available = count > 0;
    g_default_device = g_cuda_available ? DEVICE_CUDA : DEVICE_CPU;
#endif

    g_backend_initialized = 1;
    return 0;
}

int backend_initialized(void){
    return g_backend_initialized;
}

int backend_cuda_compiled(void){
#ifdef PINN_USE_CUDA
    return 1;
#else
    return 0;
#endif
}

int backend_cuda_available(void){
    if(!g_backend_initialized){
        backend_init();
    }
    return g_cuda_available;
}

Device backend_default_device(void){
    if(!g_backend_initialized){
        backend_init();
    }
    return g_default_device;
}

void backend_set_default(Device device){
    if(!g_backend_initialized){
        backend_init();
    }

    if(device == DEVICE_CUDA && !g_cuda_available){
        g_default_device = DEVICE_CPU;
        printf("Warning: CUDA is not available. Default device set to CPU.\n");
        return;
    }
    g_default_device = device;
}

int backend_device_count(void){
    if(!g_backend_initialized){
        backend_init();
    }

#ifdef PINN_USE_CUDA
    if(!g_cuda_available) return 0;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess ? count : 0;
#else
    return 0;
#endif
}

void backend_sync(Device device){
#ifdef PINN_USE_CUDA
    if(device == DEVICE_CUDA){
        cudaDeviceSynchronize();
    }
#else
    (void)device;
#endif
}

float* cuda_malloc(int n){
#ifdef PINN_USE_CUDA
    float *ptr = NULL;
    cudaError_t err = cudaMalloc((void**)&ptr, n * sizeof(float));
    if(err != cudaSuccess){
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
#else
    (void)n;
    return NULL;
#endif
}

void cuda_free(float *ptr){
#ifdef PINN_USE_CUDA
    if(ptr){
        cudaFree(ptr);
    }
#else
    (void)ptr;
#endif
}

void cuda_memcpy_to_device(const float *src, float *dst, int n){
#ifdef PINN_USE_CUDA
    cudaError_t err = cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
    }
#else
    (void)src;
    (void)dst;
    (void)n;
#endif
}

void cuda_memcpy_to_host(const float *src, float *dst, int n){
#ifdef PINN_USE_CUDA
    cudaError_t err = cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("CUDA memcpy to host failed: %s\n", cudaGetErrorString(err));
    }
#else
    (void)src;
    (void)dst;
    (void)n;
#endif
}
