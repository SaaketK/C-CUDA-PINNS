/* Reduced modal surrogate for the parametric one-dimensional heat equation. */
#ifndef PINN_SURROGATE_HEAT1D_MODAL_SURROGATE_H
#define PINN_SURROGATE_HEAT1D_MODAL_SURROGATE_H

#include "pinn/autodiff/jet.h"
#include "pinn/core/tensor.h"

enum {
    HEAT1D_MODAL_INPUT_DIM = 1,
    HEAT1D_MODAL_TAU = 0,
};

/* Largest dimensionless time alpha * (k*pi)^2 * t in the served domain. */
#define HEAT1D_MODAL_TAU_MAX (2.0f * 2.0f * 3.14159265358979323846f * 3.14159265358979323846f * 0.50f)

/*
 * The shared network learns q(tau), where tau = alpha * (k*pi)^2 * t.
 * Its hard initial-condition ansatz is
 *     q(tau) = (1 + tau * N(tau)) / (1 + tau),
 * so both spatial modes use the same dimensionless decay model, q(0) is exact,
 * and the network learns a small correction to a decaying baseline.
 */
Tensor *heat1d_modal_normalize_tau(const Tensor *tau_points);
JetTensor *heat1d_modal_jet_input(Tensor *tau_points);
Tensor *heat1d_modal_residual(JetTensor *network, Tensor *tau_points);
Tensor *heat1d_modal_coefficient(Tensor *network, Tensor *tau_points);

/* Construct tau for mode 1 or 2 from raw [t, x, alpha, a1, a2] points. */
Tensor *heat1d_modal_tau_points(const Tensor *raw_points, int mode);

/* Reconstruct u from the two modal coefficients q_1 and q_2. */
Tensor *heat1d_modal_reconstruct(
    Tensor *q1,
    Tensor *q2,
    const Tensor *raw_points
);

#endif
