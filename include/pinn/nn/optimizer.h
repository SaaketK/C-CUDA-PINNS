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

// TODO: Adam Optimizer

#endif