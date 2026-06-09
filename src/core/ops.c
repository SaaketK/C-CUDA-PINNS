/*
 * core/ops.c
 *
 * Implements tensor operation forward passes and their backward functions.
 * Each differentiable op computes output data, saves needed context, creates
 * an autograd node, and accumulates gradients into input tensors during backward.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"

Tensor* tensor_add(Tensor *a, Tensor *b){
    // Forward Pass
    Tensor* output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;

    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] + b->data[i];
    }
    return output;
}

Tensor* tensor_mult(Tensor *a, Tensor *b){
    // Forward Pass
    Tensor* output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;

    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] * b->data[i];
    }
    return output;
}

Tensor* tensor_mean(Tensor *a){
    int shape[] = {1};
    Tensor *output = tensor_create(shape, 1, a->req_grad);
    if(!output) return NULL;
    float sum = 0.0f;
    for(int i = 0; i < a->size; i++){
        sum += a->data[i];
    }
    output->data[0] = sum/a->size;
    return output;
}

Tensor* tensor_sub(Tensor *a, Tensor *b){
    Tensor* output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;

    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] - b->data[i];
    }
    return output;
}

Tensor* tensor_matmult(Tensor *a, Tensor *b){
    if(a->ndim != 2 || b->ndim != 2) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    int output_shape[] = {a->shape[0], b->shape[1]};
    Tensor *output = tensor_create(output_shape, 2, a->req_grad || b->req_grad);
    if(!output) return NULL;

    int rows = a->shape[0];
    int inner = a->shape[1];
    int cols = b->shape[1];

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float sum = 0.0f;
            for(int k = 0; k < inner; k++){
                sum += a->data[i * inner + k] * b->data[k * cols + j];
            }
            output->data[i * cols + j] = sum;
        }
    }
    return output;
}

Tensor* tensor_bias_add(Tensor *a, Tensor *b){
    if(a->ndim != 2 || b->ndim != 1) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;

    int rows = a->shape[0];
    int cols = a->shape[1];

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            output->data[i * cols + j] = a->data[i * cols + j] + b->data[j];
        }
    }

    return output;
}

Tensor* tensor_square(Tensor *a){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    if(!output) return NULL;
    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] * a->data[i];
    }
    return output;
}
Tensor* tensor_tanh(Tensor *a){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    if(!output) return NULL;
    for(int i = 0; i < a->size; i++){
        output->data[i] = tanh(a->data[i]);
    }
    return output;
}
Tensor* tensor_mse(Tensor *a, Tensor *b){
    Tensor *output = tensor_sub(a, b);
    output = tensor_mean(tensor_square(output));
    return output;
}
