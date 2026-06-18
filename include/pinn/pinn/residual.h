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

typedef struct {
    float alpha;
} Heat1DParams;

Tensor* heat1d_residual(JetTensor *u, Tensor *points, Heat1DParams *params);
Tensor* residual_mse_loss(Tensor *residual);

#endif
