/*
 * pinn/residual.h
 *
 * Owns equation callback types and residual helpers. PDE examples provide a
 * residual function that consumes model values, JetTensor derivatives, points,
 * and problem parameters while the framework stays equation-agnostic.
*/

#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "pinn/autodiff/jet.h"
#include "pinn/core/tensor.h"

typedef Tensor* (*ResidualFn)(JetTensor *u, Tensor *points, void *params);
Tensor* residual_mse_loss(Tensor *residual);

typedef struct {
    float alpha;
} Heat1DParams;

typedef struct {
    float alpha_x;
    float alpha_y;
} Heat2DParams; // Use for anisotropic diffusion

Tensor* heat1d_residual(JetTensor *u, Tensor *points, Heat1DParams *params);
Tensor* heat1d_ansatz_residual(JetTensor *N, Tensor *points, Heat1DParams *params);
Tensor* heat1d_ansatz(Tensor *raw, Tensor *points, Heat1DParams *params);
Tensor* heat2d_ansatz_residual(JetTensor *N, Tensor *points, Heat2DParams *params);
Tensor* heat2d_ansatz(Tensor *raw, Tensor *points, Heat2DParams *params);

typedef struct {
    float sigma;
    float r;
    float K;
    float T;
    float S_max;
    float payoff_beta;
} BlackScholes1DParams;

Tensor* black_scholes1d_residual(JetTensor *V, Tensor *points, BlackScholes1DParams *params);
Tensor* black_scholes1d_ansatz(Tensor *raw, Tensor *points, BlackScholes1DParams *params);

#endif
