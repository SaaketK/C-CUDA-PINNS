#include "pinn/surrogate/heat1d_modal_surrogate.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pinn/core/backend.h"
#include "pinn/core/autograd.h"
#include "pinn/core/ops.h"
#include "pinn/pinn/residual.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float *modal_tensor_to_host(const Tensor *tensor){
    float *host = malloc((size_t)tensor->size * sizeof(float));
    if(!host) return NULL;
    if(tensor->device == DEVICE_CUDA){
        cuda_memcpy_to_host(tensor->data, host, tensor->size);
    } else {
        memcpy(host, tensor->data, (size_t)tensor->size * sizeof(float));
    }
    return host;
}

static Tensor *modal_inverse_one_plus_tau(const Tensor *tau_points){
    float *tau = modal_tensor_to_host(tau_points);
    if(!tau) return NULL;
    float *inverse = malloc((size_t)tau_points->size * sizeof(float));
    if(!inverse){
        free(tau);
        return NULL;
    }
    for(int i = 0; i < tau_points->size; i++){
        inverse[i] = 1.0f / (1.0f + tau[i]);
    }
    Tensor *result = tensor_from_data_device(
        inverse,
        tau_points->shape,
        tau_points->ndim,
        0,
        tau_points->device
    );
    free(inverse);
    free(tau);
    Tape *tape = get_curr_tape();
    if(tape && result) tape_add_tensor(tape, result);
    return result;
}

Tensor *heat1d_modal_normalize_tau(const Tensor *tau_points){
    if(!tau_points || tau_points->ndim != 2
        || tau_points->shape[1] != HEAT1D_MODAL_INPUT_DIM){
        return NULL;
    }

    float *tau = modal_tensor_to_host(tau_points);
    if(!tau) return NULL;
    float *normalized = malloc((size_t)tau_points->size * sizeof(float));
    if(!normalized){
        free(tau);
        return NULL;
    }
    for(int i = 0; i < tau_points->size; i++){
        normalized[i] = 2.0f * tau[i] / HEAT1D_MODAL_TAU_MAX - 1.0f;
    }
    Tensor *result = tensor_from_data_device(
        normalized,
        tau_points->shape,
        tau_points->ndim,
        0,
        tau_points->device
    );
    free(normalized);
    free(tau);
    return result;
}

JetTensor *heat1d_modal_jet_input(Tensor *tau_points){
    Tensor *normalized = heat1d_modal_normalize_tau(tau_points);
    if(!normalized) return NULL;
    JetTensor *input = jet_create(normalized, HEAT1D_MODAL_INPUT_DIM);
    if(!input){
        tensor_free(normalized);
        return NULL;
    }

    float derivative = 2.0f / HEAT1D_MODAL_TAU_MAX;
    float *d1 = malloc((size_t)input->d1->size * sizeof(float));
    if(!d1){
        jet_free(input);
        return NULL;
    }
    for(int i = 0; i < input->d1->size; i++) d1[i] = derivative;
    if(input->d1->device == DEVICE_CUDA){
        cuda_memcpy_to_device(d1, input->d1->data, input->d1->size);
    } else {
        memcpy(input->d1->data, d1, (size_t)input->d1->size * sizeof(float));
    }
    free(d1);
    return input;
}

Tensor *heat1d_modal_residual(JetTensor *network, Tensor *tau_points){
    if(!network || !tau_points || network->input_dim != HEAT1D_MODAL_INPUT_DIM
        || network->value->size != tau_points->shape[0]){
        return NULL;
    }

    Tensor *network_tau = tensor_select_d1(
        network->d1, network->input_dim, HEAT1D_MODAL_TAU
    );
    Tensor *inverse = modal_inverse_one_plus_tau(tau_points);
    if(!network_tau || !inverse) return NULL;
    Tensor *numerator = tensor_scalar_add(
        tensor_mult(tau_points, network->value), 1.0f
    );
    Tensor *numerator_tau = tensor_add(
        network->value,
        tensor_mult(tau_points, network_tau)
    );
    Tensor *q = tensor_mult(numerator, inverse);
    Tensor *q_tau = tensor_sub(
        tensor_mult(numerator_tau, inverse),
        tensor_mult(numerator, tensor_square(inverse))
    );
    return tensor_add(q_tau, q);
}

Tensor *heat1d_modal_coefficient(Tensor *network, Tensor *tau_points){
    if(!network || !tau_points || network->size != tau_points->size
        || network->device != tau_points->device){
        return NULL;
    }
    Tensor *inverse = modal_inverse_one_plus_tau(tau_points);
    if(!inverse) return NULL;
    Tensor *numerator = tensor_scalar_add(
        tensor_mult(tau_points, network), 1.0f
    );
    return tensor_mult(numerator, inverse);
}

Tensor *heat1d_modal_tau_points(const Tensor *raw_points, int mode){
    if(!raw_points || raw_points->ndim != 2
        || raw_points->shape[1] != HEAT1D_INPUT_DIM
        || (mode != 1 && mode != 2)){
        return NULL;
    }

    float *raw = modal_tensor_to_host(raw_points);
    if(!raw) return NULL;
    int n = raw_points->shape[0];
    int shape[2] = {n, HEAT1D_MODAL_INPUT_DIM};
    float *tau = malloc((size_t)n * sizeof(float));
    if(!tau){
        free(raw);
        return NULL;
    }
    float wave_number = (float)mode * (float)M_PI;
    float wave_number_squared = wave_number * wave_number;
    for(int i = 0; i < n; i++){
        float t = raw[i * HEAT1D_INPUT_DIM + HEAT1D_T];
        float alpha = raw[i * HEAT1D_INPUT_DIM + HEAT1D_ALPHA];
        tau[i] = alpha * wave_number_squared * t;
    }
    Tensor *result = tensor_from_data_device(
        tau, shape, 2, 0, raw_points->device
    );
    free(tau);
    free(raw);
    return result;
}

Tensor *heat1d_modal_reconstruct(
    Tensor *q1,
    Tensor *q2,
    const Tensor *raw_points
){
    if(!q1 || !q2 || !raw_points || raw_points->ndim != 2
        || raw_points->shape[1] != HEAT1D_INPUT_DIM
        || q1->size != raw_points->shape[0]
        || q2->size != raw_points->shape[0]
        || q1->device != raw_points->device
        || q2->device != raw_points->device){
        return NULL;
    }

    float *raw = modal_tensor_to_host(raw_points);
    if(!raw) return NULL;
    int n = raw_points->shape[0];
    int shape[2] = {n, 1};
    float *mode1_weight = malloc((size_t)n * sizeof(float));
    float *mode2_weight = malloc((size_t)n * sizeof(float));
    if(!mode1_weight || !mode2_weight){
        free(mode1_weight);
        free(mode2_weight);
        free(raw);
        return NULL;
    }
    for(int i = 0; i < n; i++){
        float x = raw[i * HEAT1D_INPUT_DIM + HEAT1D_X];
        float a1 = raw[i * HEAT1D_INPUT_DIM + HEAT1D_A1];
        float a2 = raw[i * HEAT1D_INPUT_DIM + HEAT1D_A2];
        mode1_weight[i] = a1 * sinf((float)M_PI * x);
        mode2_weight[i] = a2 * sinf(2.0f * (float)M_PI * x);
    }
    Tensor *weight1 = tensor_from_data_device(
        mode1_weight, shape, 2, 0, raw_points->device
    );
    Tensor *weight2 = tensor_from_data_device(
        mode2_weight, shape, 2, 0, raw_points->device
    );
    free(mode1_weight);
    free(mode2_weight);
    free(raw);
    if(!weight1 || !weight2){
        tensor_free(weight1);
        tensor_free(weight2);
        return NULL;
    }

    /* The weights are constants used by the graph and share its lifetime. */
    Tape *tape = get_curr_tape();
    if(tape){
        tape_add_tensor(tape, weight1);
        tape_add_tensor(tape, weight2);
    }
    Tensor *result = tensor_add(
        tensor_mult(weight1, q1),
        tensor_mult(weight2, q2)
    );
    if(!tape){
        /* Without a tape, the operation graph does not own these constants. */
        tensor_free(weight1);
        tensor_free(weight2);
    }
    return result;
}
