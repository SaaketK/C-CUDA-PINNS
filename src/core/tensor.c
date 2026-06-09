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

Tensor* tensor_create(const int *shape, int ndim, int req_grad) {
    Tensor *tensor = (Tensor*)malloc(sizeof(Tensor));
    if(!tensor) return NULL; // Allocation failed

    tensor->ndim = ndim;
    tensor->size = tensor_size(shape, ndim);
    tensor->req_grad = req_grad;
    tensor->device = DEVICE_CPU; // Default to CPU for now
    tensor->grad_fn = NULL; // No grad_fn for leaf tensors

    // Allocate and copy shape
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if(!tensor->shape) {
        free(tensor);
        return NULL; // Allocation failed
    }
    for(int i = 0; i < ndim; i++){
        tensor->shape[i] = shape[i];
    }

    // Initialize data to zero
    tensor->data = (float*)calloc(tensor->size, sizeof(float));
    if(!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL; // Allocation failed
    }
    // Allocate grad if required
    if(req_grad){
        tensor->grad = (float*)calloc(tensor->size, sizeof(float));
        if(!tensor->grad) {
            free(tensor->data);
            free(tensor->shape);
            free(tensor);
            return NULL; // Allocation failed
        }
    }
    else {
        tensor->grad = NULL;
    }
    return tensor;
}

Tensor* tensor_from_data(const float *data, const int *shape, int ndim, int req_grad){
    if(!data) return NULL; // Invalid input

    Tensor *tensor = tensor_create(shape, ndim, req_grad);
    if(!tensor) return NULL; // Allocation failed
    
    memcpy(tensor->data, data, tensor->size * sizeof(float));
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
        if(tensor->data) free(tensor->data);
        if(tensor->grad) free(tensor->grad);
        if(tensor->shape) free(tensor->shape);
        free(tensor);
    }
}

void tensor_zero(Tensor *tensor){
    if(tensor && tensor->data){
        memset(tensor->data, 0, tensor->size * sizeof(float));
    }
}
void tensor_zero_grad(Tensor *tensor){
    if(tensor && tensor->grad){
        memset(tensor->grad, 0, tensor->size * sizeof(float));
    }
}

void tensor_fill(Tensor *tensor, float value){
    if(tensor && tensor->data){
        for(int i = 0; i < tensor->size; i++){
            tensor->data[i] = value;
        }
    }
}

void tensor_print(Tensor *tensor, const char *name){
    if(!tensor) {
        printf("Tensor %s: NULL\n", name ? name : "(unnamed)");
        return;
    }

    printf("Tensor %s: shape=[", name ? name : "(unnamed)");
    for(int i = 0; i < tensor->ndim; i++){
        printf("%d", tensor->shape[i]);
        if(i < tensor->ndim - 1) printf(", ");
    }

    printf("], data=[");
    for(int i = 0; i < tensor->size; i++){
        printf("%f", tensor->data[i]);
        if(i < tensor->size - 1) printf(", ");
    }
    printf("]\n");
}
