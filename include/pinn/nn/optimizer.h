/*
 * nn/optimizer.h
 *
 * Owns optimizer declarations: SGD, Adam, optimizer state buffers, zero_grad,
 * and parameter update APIs for trainable Tensor lists.
*/


#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "pinn/core/tensor.h"

// Stochastic Gradient Descent Optimizer

void sgd_step(Tensor **params, int n_params, float lr);
void sgd_zero_grad(Tensor **params, int n_params);

// Adam Optimizer

typedef struct {
    Tensor** params;
    Tensor** m;
    Tensor** v;
    int n_params;
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;
} Adam;

Adam* adam_create(Tensor **params, int n_params, float lr);
void adam_step(Adam *adam);
void adam_zero_grad(Adam *adam);
void adam_free(Adam *adam);

#endif
