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
Tensor* tensor_scalar_mult(Tensor *a, float scalar);
Tensor* tensor_scalar_add(Tensor *a, float scalar);
Tensor* tensor_identity(Tensor *a);
Tensor* tensor_scale_deriv(Tensor *deriv, Tensor *factor, int input_dim);
Tensor* tensor_chain_d2(Tensor *d1, Tensor *d2, Tensor *f_prime, Tensor *f_double, int input_dim);
Tensor* tensor_deriv_matmult(Tensor *deriv, Tensor *W, int batch, int in_features, int out_features, int input_dim);
Tensor* tensor_deriv2_matmult(Tensor *deriv, Tensor *W, int batch, int in_features, int out_features, int input_dim);
Tensor* tensor_select_d1(Tensor *d1, int input_dim, int component);
Tensor* tensor_select_d2(Tensor *d2, int input_dim, int p, int q);
Tensor* tensor_select_col(Tensor *a, int component);
Tensor* tensor_relu(Tensor *a);
Tensor* tensor_sigmoid(Tensor *a);

#endif
