/*
 * nn/activations.c
 *
 * Implements neural-network activation wrappers. These functions should call
 * differentiable core tensor ops such as tensor_tanh so activation layers remain
 * connected to the autograd graph.
*/

#include "pinn/nn/activations.h"
#include "pinn/core/ops.h"

Tensor *tanh_activation(Tensor *x){ return tensor_tanh(x); }
Tensor *relu_activation(Tensor *x){ return tensor_relu(x); }
Tensor *sigmoid_activation(Tensor *x){ return tensor_sigmoid(x); }
Tensor *identity_activation(Tensor *x){ return tensor_identity(x); }

