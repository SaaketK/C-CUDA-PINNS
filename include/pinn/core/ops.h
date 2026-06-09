/*
 * core/ops.h
 *
 * Declares differentiable tensor operations such as add, sub, mul, square, mean,
 * tanh, matmul, bias_add, and MSE. Operation implementations create output
 * tensors and attach autograd nodes when gradients are required.
 */

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_mult(Tensor *a, Tensor *b);
Tensor* tensor_matmult(Tensor *a, Tensor *b);
Tensor* tensor_bias_add(Tensor *a, Tensor *b);
Tensor* tensor_mse(Tensor *a, Tensor *b);
Tensor* tensor_mean(Tensor *a);
Tensor* tensor_square(Tensor *a);
Tensor* tensor_tanh(Tensor *a);


#endif