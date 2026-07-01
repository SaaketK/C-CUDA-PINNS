/*
 * nn/activations.h
 *
 * Declares neural-network activation wrappers such as tanh, and later relu or
 * sigmoid if needed. These wrappers call differentiable core tensor ops.
*/


#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "pinn/core/tensor.h"

typedef Tensor* (*ActivationFn)(Tensor *x);

Tensor *tanh_activation(Tensor *x);
Tensor *relu_activation(Tensor *x);
Tensor *sigmoid_activation(Tensor *x);
Tensor *identity_activation(Tensor *x);

#endif