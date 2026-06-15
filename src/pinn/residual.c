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

Tensor* heat1d_residual(JetTensor *u, Tensor *points, Heat1DParams *params){
    int shape[2] = {points->shape[0], 1};
    Tensor *residual = tensor_create(shape, 2, 0);
    for(int i = 0; i < shape[0]; i++){
        float u_t = jet_get_d1(u, i, 0);
        float u_xx = jet_get_d2(u, i, 1, 1);
        residual->data[i] = u_t - params->alpha * u_xx;
    }
    return residual;
}