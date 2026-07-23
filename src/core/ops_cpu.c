#include "pinn/core/ops_cpu.h"
#include <math.h>

int cpu_add(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] + b[i];
    }
    return 0;
}

int cpu_sub(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] - b[i];
    }
    return 0;
}

int cpu_mul(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] * b[i];
    }
    return 0;
}

int cpu_mean(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < size; i++){
        sum += a[i];
    }
    out[0] = sum / size;
    return 0;
}

int cpu_square(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] * a[i];
    }
    return 0;
}

int cpu_tanh(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = tanhf(a[i]);
    }
    return 0;
}

int cpu_mse(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    if(cpu_sub(a, b, out, size) != 0) return 1;
    if(cpu_square(out, out, size) != 0) return 1;
    if(cpu_mean(out, out, size) != 0) return 1;
    return 0;
}

int cpu_scalar_mult(const float *a, float scalar, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] * scalar;
    }
    return 0;
}

int cpu_scalar_add(const float *a, float scalar, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] + scalar;
    }
    return 0;
}

int cpu_identity(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i];
    }
    return 0;
}

int cpu_matmult(const float *a, const float *b, float *out, int rows, int inner, int cols){
    if(!a || !b || !out || rows < 0 || inner < 0 || cols < 0) return 1;
    if(rows * inner * cols == 0) return 0;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float sum = 0.0f;
            for(int k = 0; k < inner; k++){
                sum += a[i * inner + k] * b[k * cols + j];
            }
            out[i * cols + j] = sum;
        }
    }
    return 0;
}

int cpu_bias_add(const float *a, const float *b, float *out, int rows, int cols){
    if(!a || !b || !out || rows < 0 || cols < 0) return 1;
    if(rows * cols == 0) return 0;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            out[i * cols + j] = a[i * cols + j] + b[j];
        }
    }
    return 0;
}

int cpu_scale_deriv(const float *deriv, const float *factor, int input_dim, float *out, int size){
    if(!deriv || !factor || !out || input_dim <= 0 || size < 0 || size % input_dim != 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = deriv[i] * factor[i / input_dim];
    }
    return 0;
}

int cpu_chain_d2(const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, int size, float *out){
    if(!d1 || !d2 || !f_prime || !f_double || !out || input_dim <= 0 || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for collapse(3)
    for(int i = 0; i < size; i++){
        for(int p = 0; p < input_dim; p++){
            for(int q = 0; q < input_dim; q++){
                int idx2 = i * input_dim * input_dim + p * input_dim + q;
                int idxp = i * input_dim + p;
                int idxq = i * input_dim + q;
                out[idx2] = f_double[i] * d1[idxp] * d1[idxq] + f_prime[i] * d2[idx2];
            }
        }
    }
    return 0;
}

static int cpu_deriv_matmult_order(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, int order, float *out){
    if(!deriv || !W || !out || batch < 0 || in_features < 0 || out_features < 0 || input_dim <= 0) return 1;
    if(batch == 0 || in_features == 0 || out_features == 0) return 0;

    int channels = 1;
    for(int i = 0; i < order; i++){
        channels *= input_dim;
    }

    #pragma omp parallel for collapse(3)
    for(int r = 0; r < batch; r++){
        for(int c = 0; c < out_features; c++){
            for(int ch = 0; ch < channels; ch++){
                float sum = 0.0f;
                for(int k = 0; k < in_features; k++){
                    int deriv_idx = (r * in_features + k) * channels + ch;
                    int weight_idx = k * out_features + c;
                    sum += deriv[deriv_idx] * W[weight_idx];
                }
                int out_idx = (r * out_features + c) * channels + ch;
                out[out_idx] = sum;
            }
        }
    }
    return 0;
}

int cpu_deriv_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out){
    return cpu_deriv_matmult_order(deriv, W, batch, in_features, out_features, input_dim, 1, out);
}

int cpu_deriv2_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out){
    return cpu_deriv_matmult_order(deriv, W, batch, in_features, out_features, input_dim, 2, out);
}

int cpu_select_d1(const float *d1, int input_dim, int component, float *out, int size){
    if(!d1 || !out || input_dim <= 0 || component < 0 || component >= input_dim || size < 0 || size % input_dim != 0) return 1;
    if(size == 0) return 0;

    int out_dim = size / input_dim;
    #pragma omp parallel for
    for(int i = 0; i < out_dim; i++){
        out[i] = d1[i * input_dim + component];
    }
    return 0;
}

int cpu_select_d2(const float *d2, int input_dim, int p, int q, float *out, int size){
    if(!d2 || !out || input_dim <= 0 || p < 0 || p >= input_dim || q < 0 || q >= input_dim || size < 0 || size % (input_dim * input_dim) != 0) return 1;
    if(size == 0) return 0;

    int out_dim = size / (input_dim * input_dim);
    #pragma omp parallel for
    for(int i = 0; i < out_dim; i++){
        out[i] = d2[i * input_dim * input_dim + p * input_dim + q];
    }
    return 0;
}

int cpu_select_col(const float *a, int component, float *out, int rows, int cols){
    if(!a || !out || rows < 0 || cols <= 0 || component < 0 || component >= cols) return 1;
    if(rows == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < rows; i++){
        out[i] = a[i * cols + component];
    }
    return 0;
}

int cpu_relu(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
    return 0;
}

int cpu_sigmoid(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        out[i] = 1.0f / (1.0f + exp(-1.0f * a[i]));
    }
    return 0;
}
