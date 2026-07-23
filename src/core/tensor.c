/*
 * core/tensor.c
 *
 * Implements Tensor memory management: allocating tensors, copying shape/data,
 * allocating gradient buffers, freeing owned memory, zeroing/filling buffers,
 * and printing tensors for debugging.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pinn/core/tensor.h"
#include "pinn/core/backend.h"

static int tensor_uses_cuda(Device device){
    return device == DEVICE_CUDA && backend_cuda_available();
}

static void tensor_free_buffer(float *ptr, Device device){
    if(!ptr) return;
    if(tensor_uses_cuda(device)){
        cuda_free(ptr);
    }
    else {
        free(ptr);
    }
}

static float* tensor_alloc_zeroed(int n, Device device){
    if(tensor_uses_cuda(device)){
        float *ptr = cuda_malloc(n);
        if(!ptr) return NULL;
        float *zeros = calloc(n, sizeof(float));
        if(!zeros){
            cuda_free(ptr);
            return NULL;
        }
        cuda_memcpy_to_device(zeros, ptr, n);
        free(zeros);
        return ptr;
    }

    return (float*)calloc(n, sizeof(float));
}

static void tensor_copy_from_host(float *dst, const float *src, int n, Device device){
    if(tensor_uses_cuda(device)){
        cuda_memcpy_to_device(src, dst, n);
    }
    else {
        memcpy(dst, src, n * sizeof(float));
    }
}

static void tensor_copy_to_host(float *dst, const float *src, int n, Device device){
    if(tensor_uses_cuda(device)){
        cuda_memcpy_to_host(src, dst, n);
    }
    else {
        memcpy(dst, src, n * sizeof(float));
    }
}

Tensor* tensor_create(const int *shape, int ndim, int req_grad){
    return tensor_create_device(shape, ndim, req_grad, DEVICE_CPU);
}

Tensor* tensor_create_device(const int *shape, int ndim, int req_grad, Device device) {
    Tensor *tensor = (Tensor*)malloc(sizeof(Tensor));
    if(!tensor) return NULL;

    if(device == DEVICE_CUDA && !backend_cuda_available()){
        printf("Warning: CUDA requested but unavailable; using CPU tensor instead.\n");
        device = DEVICE_CPU;
    }

    tensor->ndim = ndim;
    tensor->size = tensor_size(shape, ndim);
    tensor->req_grad = req_grad;
    tensor->device = device;
    tensor->grad_fn = NULL;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if(!tensor->shape) {
        free(tensor);
        return NULL;
    }
    for(int i = 0; i < ndim; i++){
        tensor->shape[i] = shape[i];
    }

    tensor->data = tensor_alloc_zeroed(tensor->size, tensor->device);
    if(!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    if(req_grad){
        tensor->grad = tensor_alloc_zeroed(tensor->size, tensor->device);
        if(!tensor->grad) {
            tensor_free_buffer(tensor->data, tensor->device);
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
    }
    else {
        tensor->grad = NULL;
    }
    return tensor;
}

Tensor* tensor_from_data(const float *data, const int *shape, int ndim, int req_grad){
    return tensor_from_data_device(data, shape, ndim, req_grad, DEVICE_CPU);
}

Tensor* tensor_from_data_device(const float *data, const int *shape, int ndim, int req_grad, Device device){
    if(!data) return NULL;

    Tensor *tensor = tensor_create_device(shape, ndim, req_grad, device);
    if(!tensor) return NULL;

    tensor_copy_from_host(tensor->data, data, tensor->size, tensor->device);
    return tensor;
}

int tensor_size(const int *shape, int ndim){
    int size = 1;
    for(int i = 0; i < ndim; i++){
        size *= shape[i];
    }
    return size;
}

void tensor_free(Tensor *tensor){
    if(tensor){
        tensor_free_buffer(tensor->data, tensor->device);
        tensor_free_buffer(tensor->grad, tensor->device);
        if(tensor->shape) free(tensor->shape);
        free(tensor);
    }
}

void tensor_zero(Tensor *tensor){
    if(tensor && tensor->data){
        if(tensor_uses_cuda(tensor->device)){
            float *zeros = calloc(tensor->size, sizeof(float));
            if(!zeros) return;
            cuda_memcpy_to_device(zeros, tensor->data, tensor->size);
            free(zeros);
        }
        else {
            memset(tensor->data, 0, tensor->size * sizeof(float));
        }
    }
}

void tensor_zero_grad(Tensor *tensor){
    if(tensor && tensor->grad){
        if(tensor_uses_cuda(tensor->device)){
            float *zeros = calloc(tensor->size, sizeof(float));
            if(!zeros) return;
            cuda_memcpy_to_device(zeros, tensor->grad, tensor->size);
            free(zeros);
        }
        else {
            memset(tensor->grad, 0, tensor->size * sizeof(float));
        }
    }
}

void tensor_fill(Tensor *tensor, float value){
    if(tensor && tensor->data){
        if(tensor_uses_cuda(tensor->device)){
            float *values = (float*)malloc(tensor->size * sizeof(float));
            if(!values) return;
            for(int i = 0; i < tensor->size; i++){
                values[i] = value;
            }
            cuda_memcpy_to_device(values, tensor->data, tensor->size);
            free(values);
        }
        else {
            for(int i = 0; i < tensor->size; i++){
                tensor->data[i] = value;
            }
        }
    }
}

void tensor_print(Tensor *tensor, const char *name){
    if(!tensor) {
        printf("Tensor %s: NULL\n", name ? name : "(unnamed)");
        return;
    }

    float *host_data = tensor->data;
    float *scratch = NULL;
    if(tensor_uses_cuda(tensor->device)){
        scratch = (float*)malloc(tensor->size * sizeof(float));
        if(!scratch){
            printf("Tensor %s: unable to copy from CUDA device\n", name ? name : "(unnamed)");
            return;
        }
        tensor_copy_to_host(scratch, tensor->data, tensor->size, tensor->device);
        host_data = scratch;
    }

    printf("Tensor %s: shape=[", name ? name : "(unnamed)");
    for(int i = 0; i < tensor->ndim; i++){
        printf("%d", tensor->shape[i]);
        if(i < tensor->ndim - 1) printf(", ");
    }

    printf("], data=[");
    for(int i = 0; i < tensor->size; i++){
        printf("%f", host_data[i]);
        if(i < tensor->size - 1) printf(", ");
    }
    printf("]\n");

    free(scratch);
}

void tensor_to_cpu(Tensor *tensor){
    if(!tensor || tensor->device == DEVICE_CPU) return;
    if(!tensor_uses_cuda(tensor->device)){
        tensor->device = DEVICE_CPU;
        return;
    }

    float *new_data = (float*)calloc(tensor->size, sizeof(float));
    if(!new_data) return;
    tensor_copy_to_host(new_data, tensor->data, tensor->size, tensor->device);

    float *new_grad = NULL;
    if(tensor->grad){
        new_grad = (float*)calloc(tensor->size, sizeof(float));
        if(!new_grad){
            free(new_data);
            return;
        }
        tensor_copy_to_host(new_grad, tensor->grad, tensor->size, tensor->device);
    }

    tensor_free_buffer(tensor->data, tensor->device);
    tensor_free_buffer(tensor->grad, tensor->device);
    tensor->data = new_data;
    tensor->grad = new_grad;
    tensor->device = DEVICE_CPU;
}

void tensor_to_cuda(Tensor *tensor){
    if(!tensor || tensor->device == DEVICE_CUDA) return;
    if(!backend_cuda_available()) return;

    float *new_data = cuda_malloc(tensor->size);
    if(!new_data) return;
    tensor_copy_from_host(new_data, tensor->data, tensor->size, DEVICE_CUDA);

    float *new_grad = NULL;
    if(tensor->grad){
        new_grad = cuda_malloc(tensor->size);
        if(!new_grad){
            cuda_free(new_data);
            return;
        }
        tensor_copy_from_host(new_grad, tensor->grad, tensor->size, DEVICE_CUDA);
    }

    tensor_free_buffer(tensor->data, tensor->device);
    tensor_free_buffer(tensor->grad, tensor->device);
    tensor->data = new_data;
    tensor->grad = new_grad;
    tensor->device = DEVICE_CUDA;
}
