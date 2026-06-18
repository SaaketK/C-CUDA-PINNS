/*
 * pinn/residual.c
 *
 * Will implement equation-agnostic residual helpers and constraint-loss
 * utilities. Equation-specific callbacks, such as heat or Black-Scholes
 * residuals, will use JetTensor derivatives to form physics losses while the
 * trainer handles sampling and optimization.
*/

#include <stdio.h>
#include <stdlib.h>
#include "pinn/pinn/residual.h"
#include "pinn/autodiff/jet.h"
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"

Tensor* heat1d_residual(JetTensor *u, Tensor *points, Heat1DParams *params){
    (void)points;
    Tensor *u_t = tensor_select_d1(u->d1, u->input_dim, 0);
    Tensor *u_xx = tensor_select_d2(u->d2, u->input_dim, 1, 1);
    Tensor *alpha_u_xx = tensor_scalar_mult(u_xx, params->alpha);
    Tensor *residual = tensor_sub(u_t, alpha_u_xx);
    return residual;
}

Tensor* residual_mse_loss(Tensor *residual){
    Tensor *square = tensor_square(residual);
    return tensor_mean(square);
}