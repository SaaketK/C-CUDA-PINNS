/* Normalization and input-contract helpers for the Heat 1D surrogate. */
#ifndef PINN_SURROGATE_HEAT1D_SURROGATE_H
#define PINN_SURROGATE_HEAT1D_SURROGATE_H

#include "pinn/core/tensor.h"
#include "pinn/autodiff/jet.h"

enum {
    HEAT1D_COEFF_T = 0,
    HEAT1D_COEFF_X = 1,
    HEAT1D_COEFF_ALPHA = 2,
    HEAT1D_COEFF_SIN_PI_X = 3,
    HEAT1D_COEFF_COS_PI_X = 4,
    HEAT1D_COEFF_SIN_2PI_X = 5,
    HEAT1D_COEFF_COS_2PI_X = 6,
    HEAT1D_COEFFICIENT_INPUT_DIM = 7,
    HEAT1D_COEFFICIENT_PHYSICAL_DIM = 3,
};

/*
 * Converts raw [t, x, alpha, a1, a2] inputs to the model representation.
 * t, x, alpha, a1 and a2 are each mapped from their served ranges to [-1, 1].
 * The returned tensor has the same device as raw_points and must be freed by 
 * the caller. PDE residuals must still be formed with the raw coordinates.
 */
Tensor *heat1d_normalize_inputs(const Tensor *raw_points);
JetTensor *heat1d_normalized_jet_input(Tensor *raw_points);

/*
 * Feature contract for the independent Fourier-mode coefficient models:
 * [t_n, x_n, alpha_n, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x)].
 * The JetTensor carries derivatives with respect to raw [t, x, alpha], not
 * with respect to the seven derived features.
 */
Tensor *heat1d_coefficient_features(const Tensor *raw_points);
JetTensor *heat1d_coefficient_feature_jet_input(Tensor *raw_points);

#endif
