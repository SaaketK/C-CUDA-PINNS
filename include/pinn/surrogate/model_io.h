#ifndef MODEL_IO_H
#define MODEL_IO_H

#include "pinn/nn/mlp.h"
#include "pinn/nn/activations.h"

int pinn_model_save(const char *path, const MLP *mlp, const float *input_lower, const float *input_upper);
MLP* pinn_model_load(const char *path, float **input_lower_out, float **input_upper_out, ActivationFn activation_fn);

#endif