/*
 * nn/optimizer.c
 *
 * Implements optimizers for trainable tensor parameters. Start with SGD for
 * simple parameter updates, then add Adam state buffers and update rules once
 * the MLP training loop is working.
*/

#include <stdio.h>
#include <stdlib.h>
#include "pinn/nn/optimizer.h"
#include "pinn/core/tensor.h"

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