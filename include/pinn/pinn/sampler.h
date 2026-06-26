/*
 * pinn/sampler.h
 *
 * Owns collocation and constraint sampling interfaces: uniform sampling, LHS,
 * boundary/terminal point sampling, and later adaptive or Sinkhorn-based
 * sampler update hooks.
*/

#ifndef SAMPLER_H
#define SAMPLER_H

#include "pinn/core/tensor.h"

typedef struct {
    int dim;
    float *lower;
    float *upper;
} BoxDomain;

Tensor* sample_uniform_box(BoxDomain *domain, int n_points);
Tensor* sample_fixed_dim_box(BoxDomain *domain, int n_points, int fixed_dim, float fixed_value);
Tensor* sample_LHS_box(BoxDomain *domain, int n);
 
#endif
