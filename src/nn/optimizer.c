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
#include "pinn/core/tensor.h"

// Stochastic Gradient Descent

void sgd_step(Tensor **params, int n_params, float lr){
    // new weight = old weight - learning_rate * gradient
    for(int i = 0; i < n_params; i++){
        for(int j = 0; j < params[i]->size; j++){
            params[i]->data[j] = params[i]->data[j] - lr * params[i]->grad[j];
        }
    } 
}

void sgd_zero_grad(Tensor **params, int n_params){
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
    for(int i = 0; i < n_params; i++){
        adam->params[i] = params[i];
        adam->m[i] = tensor_create(params[i]->shape, params[i]->ndim, 0);
        adam->v[i] = tensor_create(params[i]->shape, params[i]->ndim, 0);
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
        for(int j = 0; j < adam->params[i]->size; j++){
            float g = adam->params[i]->grad[j];
            adam->m[i]->data[j] = adam->beta1 * adam->m[i]->data[j] + (1 - adam->beta1) * g;
            adam->v[i]->data[j] = adam->beta2 * adam->v[i]->data[j] + (1 - adam->beta2) * g * g;
            float m_hat = adam->m[i]->data[j] / beta1_correction;
            float v_hat = adam->v[i]->data[j] / beta2_correction;
            adam->params[i]->data[j] = adam->params[i]->data[j] - adam->lr * m_hat / (sqrtf(v_hat) + adam->eps);
        }
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
