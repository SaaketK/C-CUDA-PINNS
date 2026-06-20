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

Tensor* black_scholes1d_residual(JetTensor *V, Tensor *points, BlackScholes1DParams *params){
    // V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0
    Tensor *V_t = tensor_select_d1(V->d1, V->input_dim, 0);
    Tensor *V_S = tensor_select_d1(V->d1, V->input_dim, 1);
    Tensor *V_SS = tensor_select_d2(V->d2, V->input_dim, 1, 1);
    Tensor *S = tensor_select_col(points, 1);
    // let w = 0.5 * sigma^2 * S^2 
    Tensor *w = tensor_scalar_mult(tensor_scalar_mult(tensor_square(S), params->sigma * params->sigma), 0.5f);
    Tensor *w_V_SS = tensor_mult(w, V_SS);
    Tensor *r_S_V_S = tensor_scalar_mult(tensor_mult(S, V_S), params->r);
    Tensor *r_V = tensor_scalar_mult(V->value, params->r);
    Tensor *residual = tensor_add(tensor_add(V_t, w_V_SS), tensor_sub(r_S_V_S,r_V));
    return residual;
}