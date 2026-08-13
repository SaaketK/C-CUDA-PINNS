#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pinn/core/backend.h"
#include "pinn/core/tensor.h"
#include "pinn/pinn/residual.h"
#include "pinn/surrogate/heat1d_surrogate.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float *heat1d_points_to_host(const Tensor *raw_points){
    float *raw = malloc((size_t)raw_points->size * sizeof(float));
    if(!raw) return NULL;
    if(raw_points->device == DEVICE_CUDA){
        cuda_memcpy_to_host(raw_points->data, raw, raw_points->size);
    } else {
        memcpy(raw, raw_points->data, (size_t)raw_points->size * sizeof(float));
    }
    return raw;
}

Tensor *heat1d_normalize_inputs(const Tensor *raw_points){
    if(!raw_points || raw_points->ndim != 2
        || raw_points->shape[1] != HEAT1D_INPUT_DIM){
        return NULL;
    }

    int shape[2] = {raw_points->shape[0], HEAT1D_INPUT_DIM};
    Tensor *normalized = tensor_create_device(shape, 2, 0, raw_points->device);
    if(!normalized) return NULL;

    float *raw = heat1d_points_to_host(raw_points);
    float *values = malloc((size_t)raw_points->size * sizeof(float));
    if(!raw || !values){
        free(raw);
        free(values);
        tensor_free(normalized);
        return NULL;
    }
    for(int i = 0; i < shape[0]; i++){
        int offset = i * HEAT1D_INPUT_DIM;
        values[offset + HEAT1D_T] = 2.0f * raw[offset + HEAT1D_T] - 1.0f;
        values[offset + HEAT1D_X] = 2.0f * raw[offset + HEAT1D_X] - 1.0f;
        values[offset + HEAT1D_ALPHA] =
            2.0f * (raw[offset + HEAT1D_ALPHA] - 0.01f) / 0.49f - 1.0f;
        values[offset + HEAT1D_A1] = raw[offset + HEAT1D_A1];
        values[offset + HEAT1D_A2] = raw[offset + HEAT1D_A2];
    }

    if(normalized->device == DEVICE_CUDA){
        cuda_memcpy_to_device(values, normalized->data, normalized->size);
    } else {
        for(int i = 0; i < normalized->size; i++) normalized->data[i] = values[i];
    }
    free(raw);
    free(values);
    return normalized;
}

Tensor *heat1d_coefficient_features(const Tensor *raw_points){
    if(!raw_points || raw_points->ndim != 2
        || raw_points->shape[1] != HEAT1D_INPUT_DIM){
        return NULL;
    }

    float *raw = heat1d_points_to_host(raw_points);
    if(!raw) return NULL;
    int n = raw_points->shape[0];
    int shape[2] = {n, HEAT1D_COEFFICIENT_INPUT_DIM};
    float *features = malloc(
        (size_t)n * HEAT1D_COEFFICIENT_INPUT_DIM * sizeof(float)
    );
    if(!features){
        free(raw);
        return NULL;
    }

    for(int i = 0; i < n; i++){
        int raw_offset = i * HEAT1D_INPUT_DIM;
        int feature_offset = i * HEAT1D_COEFFICIENT_INPUT_DIM;
        float t = raw[raw_offset + HEAT1D_T];
        float x = raw[raw_offset + HEAT1D_X];
        float alpha = raw[raw_offset + HEAT1D_ALPHA];
        float pi_x = (float)M_PI * x;
        features[feature_offset + HEAT1D_COEFF_T] = 2.0f * t - 1.0f;
        features[feature_offset + HEAT1D_COEFF_X] = 2.0f * x - 1.0f;
        features[feature_offset + HEAT1D_COEFF_ALPHA] =
            2.0f * (alpha - 0.01f) / 0.49f - 1.0f;
        features[feature_offset + HEAT1D_COEFF_SIN_PI_X] = sinf(pi_x);
        features[feature_offset + HEAT1D_COEFF_COS_PI_X] = cosf(pi_x);
        features[feature_offset + HEAT1D_COEFF_SIN_2PI_X] = sinf(2.0f * pi_x);
        features[feature_offset + HEAT1D_COEFF_COS_2PI_X] = cosf(2.0f * pi_x);
    }

    Tensor *result = tensor_from_data_device(
        features, shape, 2, 0, raw_points->device
    );
    free(features);
    free(raw);
    return result;
}

JetTensor *heat1d_coefficient_feature_jet_input(Tensor *raw_points){
    Tensor *features = heat1d_coefficient_features(raw_points);
    if(!features) return NULL;
    JetTensor *input = jet_create(features, HEAT1D_COEFFICIENT_PHYSICAL_DIM);
    if(!input){
        tensor_free(features);
        return NULL;
    }

    int n = raw_points->shape[0];
    int d1_size = input->d1->size;
    int d2_size = input->d2->size;
    float *d1 = calloc((size_t)d1_size, sizeof(float));
    float *d2 = calloc((size_t)d2_size, sizeof(float));
    float *raw = heat1d_points_to_host(raw_points);
    if(!d1 || !d2 || !raw){
        free(d1);
        free(d2);
        free(raw);
        jet_free(input);
        return NULL;
    }

    const float pi = (float)M_PI;
    const float pi2 = pi * pi;
    for(int row = 0; row < n; row++){
        float x = raw[row * HEAT1D_INPUT_DIM + HEAT1D_X];
        float sin1 = sinf(pi * x);
        float cos1 = cosf(pi * x);
        float sin2 = sinf(2.0f * pi * x);
        float cos2 = cosf(2.0f * pi * x);
        int base = row * HEAT1D_COEFFICIENT_INPUT_DIM;

#define D1(feature, physical) \
        d1[((base + (feature)) * HEAT1D_COEFFICIENT_PHYSICAL_DIM) + (physical)]
#define D2(feature, first, second) \
        d2[(((base + (feature)) * HEAT1D_COEFFICIENT_PHYSICAL_DIM \
            + (first)) * HEAT1D_COEFFICIENT_PHYSICAL_DIM) + (second)]
        D1(HEAT1D_COEFF_T, HEAT1D_T) = 2.0f;
        D1(HEAT1D_COEFF_X, HEAT1D_X) = 2.0f;
        D1(HEAT1D_COEFF_ALPHA, HEAT1D_ALPHA) = 2.0f / 0.49f;
        D1(HEAT1D_COEFF_SIN_PI_X, HEAT1D_X) = pi * cos1;
        D1(HEAT1D_COEFF_COS_PI_X, HEAT1D_X) = -pi * sin1;
        D1(HEAT1D_COEFF_SIN_2PI_X, HEAT1D_X) = 2.0f * pi * cos2;
        D1(HEAT1D_COEFF_COS_2PI_X, HEAT1D_X) = -2.0f * pi * sin2;
        D2(HEAT1D_COEFF_SIN_PI_X, HEAT1D_X, HEAT1D_X) = -pi2 * sin1;
        D2(HEAT1D_COEFF_COS_PI_X, HEAT1D_X, HEAT1D_X) = -pi2 * cos1;
        D2(HEAT1D_COEFF_SIN_2PI_X, HEAT1D_X, HEAT1D_X) = -4.0f * pi2 * sin2;
        D2(HEAT1D_COEFF_COS_2PI_X, HEAT1D_X, HEAT1D_X) = -4.0f * pi2 * cos2;
#undef D1
#undef D2
    }

    if(features->device == DEVICE_CUDA){
        cuda_memcpy_to_device(d1, input->d1->data, d1_size);
        cuda_memcpy_to_device(d2, input->d2->data, d2_size);
    } else {
        memcpy(input->d1->data, d1, (size_t)d1_size * sizeof(float));
        memcpy(input->d2->data, d2, (size_t)d2_size * sizeof(float));
    }
    free(d1);
    free(d2);
    free(raw);
    return input;
}


JetTensor *heat1d_normalized_jet_input(Tensor *raw_points){
    Tensor *normalized = heat1d_normalize_inputs(raw_points);
    if(!normalized) return NULL;
    JetTensor *input = jet_create_input(normalized, HEAT1D_INPUT_DIM);
    if(!input){
        tensor_free(normalized);
        return NULL;
    }

    /* Map jet derivatives back to the raw t and x coordinates. */
    float *d1 = malloc((size_t)input->d1->size * sizeof(float));
    if(!d1){
        jet_free(input);
        return NULL;
    }
    if(input->d1->device == DEVICE_CUDA){
        cuda_memcpy_to_host(input->d1->data, d1, input->d1->size);
    } else {
        for(int i = 0; i < input->d1->size; i++) d1[i] = input->d1->data[i];
    }
    int batch = raw_points->shape[0];
    for(int i = 0; i < batch; i++){
        d1[(i * HEAT1D_INPUT_DIM + HEAT1D_T) * HEAT1D_INPUT_DIM + HEAT1D_T] = 2.0f;
        d1[(i * HEAT1D_INPUT_DIM + HEAT1D_X) * HEAT1D_INPUT_DIM + HEAT1D_X] = 2.0f;
    }
    if(input->d1->device == DEVICE_CUDA){
        cuda_memcpy_to_device(d1, input->d1->data, input->d1->size);
    } else {
        for(int i = 0; i < input->d1->size; i++)input->d1->data[i] = d1[i];
    }
    free(d1);
    return input;
}
