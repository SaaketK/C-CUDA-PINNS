/*
 * nn/optimizer.c
 *
 * Implements optimizers for trainable tensor parameters. Start with SGD for
 * simple parameter updates, then add Adam state buffers and update rules once
 * the MLP training loop is working.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pinn/nn/optimizer.h"
#include "pinn/nn/optimizer_cpu.h"
#include "pinn/core/tensor.h"

#ifdef PINN_USE_CUDA
    #include "pinn/nn/optimizer_cuda.h"
#endif

// Stochastic Gradient Descent

void sgd_step(Tensor **params, int n_params, float lr){
    // new weight = old weight - learning_rate * gradient
    for(int i = 0; i < n_params; i++){
#ifdef PINN_USE_CUDA
        if(params[i]->device == DEVICE_CUDA){
            cuda_sgd_step(params[i]->data, params[i]->grad, lr, params[i]->size);
            continue;
        }
#endif
        cpu_sgd_step(params[i]->data, params[i]->grad, lr, params[i]->size);
    } 
}

void sgd_zero_grad(Tensor **params, int n_params){
    #pragma omp parallel for
    for(int i = 0; i < n_params; i++){
        tensor_zero_grad(params[i]);
    }
}

// Adam

static float beta1 = 0.9f;
static float beta2 = 0.999f;
static float eps = 1e-8f;

Adam* adam_create(Tensor **params, int n_params, float lr){
    Adam *adam = malloc(sizeof(Adam));
    adam->params = malloc(n_params * sizeof(Tensor*));
    adam->m = calloc(n_params, sizeof(Tensor*));
    adam->v = calloc(n_params, sizeof(Tensor*));
    adam->n_params = n_params;
    #pragma omp parallel for
    for(int i = 0; i < n_params; i++){
        adam->params[i] = params[i];
        adam->m[i] = tensor_create_device(params[i]->shape, params[i]->ndim, 0, params[i]->device);
        adam->v[i] = tensor_create_device(params[i]->shape, params[i]->ndim, 0, params[i]->device);
    }
    adam->lr = lr;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->eps = eps;
    adam->t = 0;
    return adam;
}

void adam_step(Adam *adam){
    adam->t++;
    float beta1_correction = 1.0f - powf(adam->beta1, adam->t);
    float beta2_correction = 1.0f - powf(adam->beta2, adam->t);

    for(int i = 0; i < adam->n_params; i++){
#ifdef PINN_USE_CUDA
        if(adam->params[i]->device == DEVICE_CUDA){
            cuda_adam_step(adam->params[i]->data, adam->params[i]->grad, adam->m[i]->data, adam->v[i]->data, adam->lr, adam->beta1, adam->beta2, adam->eps, beta1_correction, beta2_correction, adam->params[i]->size);
            continue;
        }
#endif
        cpu_adam_step(
            adam->params[i]->data, adam->params[i]->grad,
            adam->m[i]->data, adam->v[i]->data,
            adam->lr, adam->beta1, adam->beta2, adam->eps,
            beta1_correction, beta2_correction, adam->params[i]->size
        );
    }
}

void adam_zero_grad(Adam *adam){
    for(int i = 0; i < adam->n_params; i++){
        tensor_zero_grad(adam->params[i]);
    }
}

void adam_free(Adam *adam){
    for(int i = 0;i < adam->n_params; i++){
        tensor_free(adam->m[i]);
        tensor_free(adam->v[i]);
    }
    free(adam->params);
    free(adam->m);
    free(adam->v);
    free(adam);
}
