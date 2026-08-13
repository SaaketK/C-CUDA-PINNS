#include "pinn/core/ops_cpu.h"
#include <math.h>

#ifdef PINN_USE_OPENBLAS
    #include <cblas.h>
#endif

// Forward Operations with OpenMP parallelization

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

#ifdef PINN_USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, cols, inner, 1.0f, a, inner, b, cols, 0.0f, out, cols);
#else
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
#endif
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

#ifdef PINN_USE_OPENBLAS
    for(int r = 0; r < batch; r++){
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    out_features, channels, in_features, 1.0f,
                    W, out_features,
                    deriv + r * in_features * channels, channels,
                    0.0f, out + r * out_features * channels, channels);
    }
#else
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
#endif
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

// Backward

int cpu_add_backward(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i];
        if(b_req_grad) b_grad[i] += out_grad[i];
    }
    return 0;
}

int cpu_sub_backward(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i];
        if(b_req_grad) b_grad[i] -= out_grad[i];
    }
    return 0;
}

int cpu_mult_backward(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    if(!a || !b || !out_grad || size < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i] * b[i];
        if(b_req_grad) b_grad[i] += out_grad[i] * a[i];
    }
    return 0;
}

int cpu_matmult_backward(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int rows, int inner, int cols, int a_req_grad, int b_req_grad){
    if(!a || !b || !out_grad || rows < 0 || inner < 0 || cols < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;

    if(a_req_grad){
#ifdef PINN_USE_OPENBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, inner, cols, 1.0f, out_grad, cols, b, cols, 1.0f, a_grad, inner);
#else
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < rows; i++){
            for(int k = 0; k < inner; k++){
                float sum = 0.0f;
                for(int j = 0; j < cols; j++) sum += out_grad[i * cols + j] * b[k * cols + j];
                a_grad[i * inner + k] += sum;
            }
        }
#endif
    }
    if(b_req_grad){
#ifdef PINN_USE_OPENBLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    inner, cols, rows, 1.0f, a, inner, out_grad, cols, 1.0f, b_grad, cols);
#else
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < inner; k++){
            for(int j = 0; j < cols; j++){
                float sum = 0.0f;
                for(int i = 0; i < rows; i++) sum += out_grad[i * cols + j] * a[i * inner + k];
                b_grad[k * cols + j] += sum;
            }
        }
#endif
    }
    return 0;
}

int cpu_bias_add_backward(const float *out_grad, float *a_grad, float *b_grad, int rows, int cols, int a_req_grad, int b_req_grad){
    if(!out_grad || rows < 0 || cols < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;

    if(a_req_grad){
        #pragma omp parallel for
        for(int i = 0; i < rows * cols; i++) a_grad[i] += out_grad[i];
    }
    if(b_req_grad){
        #pragma omp parallel for
        for(int j = 0; j < cols; j++){
            float sum = 0.0f;
            for(int i = 0; i < rows; i++) sum += out_grad[i * cols + j];
            b_grad[j] += sum;
        }
    }
    return 0;
}

int cpu_mean_backward(const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[0] / size;
    }
    return 0;
}

int cpu_square_backward(const float *a, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!a || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += 2.0f * out_grad[i] * a[i];
    }
    return 0;
}

int cpu_tanh_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i] * (1.0f - out[i] * out[i]);
    }
    return 0;
}

int cpu_scalar_mult_backward(const float *out_grad, float scalar, float *a_grad, int size, int a_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i] * scalar;
    }
    return 0;
}

int cpu_scalar_add_backward(const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i];
    }
    return 0;
}

int cpu_identity_backward(const float *out_grad, float *a_grad, int size, int a_req_grad){
    return cpu_scalar_add_backward(out_grad, a_grad, size, a_req_grad);
}

int cpu_scale_deriv_backward(const float *deriv, const float *factor, const float *out_grad, float *deriv_grad, float *factor_grad, int input_dim, int size, int deriv_req_grad, int factor_req_grad){
    if(!deriv || !factor || !out_grad || input_dim <= 0 || size < 0 || size % input_dim != 0 || (deriv_req_grad && !deriv_grad) || (factor_req_grad && !factor_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size / input_dim; i++){
        float factor_sum = 0.0f;
        for(int j = 0; j < input_dim; j++){
            int idx = i * input_dim + j;
            if(deriv_req_grad) deriv_grad[idx] += out_grad[idx] * factor[i];
            if(factor_req_grad) factor_sum += out_grad[idx] * deriv[idx];
        }
        if(factor_req_grad) factor_grad[i] += factor_sum;
    }
    return 0;
}

int cpu_chain_d2_backward(const float *d1, const float *d2, const float *f_prime, const float *f_double, const float *out_grad, float *d1_grad, float *d2_grad, float *f_prime_grad, float *f_double_grad, int input_dim, int size, int d1_req_grad, int d2_req_grad, int f_prime_req_grad, int f_double_req_grad){
    if(!d1 || !d2 || !f_prime || !f_double || !out_grad || input_dim <= 0 || size < 0 || (d1_req_grad && !d1_grad) || (d2_req_grad && !d2_grad) || (f_prime_req_grad && !f_prime_grad) || (f_double_req_grad && !f_double_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        float f_prime_sum = 0.0f;
        float f_double_sum = 0.0f;
        for(int p = 0; p < input_dim; p++){
            for(int q = 0; q < input_dim; q++){
                int idx2 = i * input_dim * input_dim + p * input_dim + q;
                int idxp = i * input_dim + p;
                int idxq = i * input_dim + q;
                float grad = out_grad[idx2];
                if(d1_req_grad){
                    d1_grad[idxp] += grad * f_double[i] * d1[idxq];
                    d1_grad[idxq] += grad * f_double[i] * d1[idxp];
                }
                if(d2_req_grad) d2_grad[idx2] += grad * f_prime[i];
                if(f_prime_req_grad) f_prime_sum += grad * d2[idx2];
                if(f_double_req_grad) f_double_sum += grad * d1[idxp] * d1[idxq];
            }
        }
        if(f_prime_req_grad) f_prime_grad[i] += f_prime_sum;
        if(f_double_req_grad) f_double_grad[i] += f_double_sum;
    }
    return 0;
}

int cpu_deriv_matmult_backward(const float *deriv, const float *W, const float *out_grad, float *deriv_grad, float *W_grad, int batch, int in_features, int out_features, int input_dim, int order, int deriv_req_grad, int W_req_grad){
    if(!deriv || !W || !out_grad || batch < 0 || in_features < 0 || out_features < 0 || input_dim <= 0 || order <= 0 || (deriv_req_grad && !deriv_grad) || (W_req_grad && !W_grad)) return 1;

    int channels = 1;
    for(int i = 0; i < order; i++) channels *= input_dim;

    if(deriv_req_grad){
#ifdef PINN_USE_OPENBLAS
        for(int r = 0; r < batch; r++){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        in_features, channels, out_features, 1.0f,
                        W, out_features,
                        out_grad + r * out_features * channels, channels,
                        1.0f, deriv_grad + r * in_features * channels, channels);
        }
#else
        #pragma omp parallel for collapse(3)
        for(int r = 0; r < batch; r++){
            for(int k = 0; k < in_features; k++){
                for(int ch = 0; ch < channels; ch++){
                    float sum = 0.0f;
                    for(int c = 0; c < out_features; c++) sum += out_grad[(r * out_features + c) * channels + ch] * W[k * out_features + c];
                    deriv_grad[(r * in_features + k) * channels + ch] += sum;
                }
            }
        }
#endif
    }
    if(W_req_grad){
#ifdef PINN_USE_OPENBLAS
        for(int r = 0; r < batch; r++){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        in_features, out_features, channels, 1.0f,
                        deriv + r * in_features * channels, channels,
                        out_grad + r * out_features * channels, channels,
                        1.0f, W_grad, out_features);
        }
#else
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < in_features; k++){
            for(int c = 0; c < out_features; c++){
                float sum = 0.0f;
                for(int r = 0; r < batch; r++){
                    for(int ch = 0; ch < channels; ch++) sum += out_grad[(r * out_features + c) * channels + ch] * deriv[(r * in_features + k) * channels + ch];
                }
                W_grad[k * out_features + c] += sum;
            }
        }
#endif
    }
    return 0;
}

int cpu_select_d1_backward(const float *out_grad, float *d1_grad, int input_dim, int component, int size, int d1_req_grad){
    if(!out_grad || input_dim <= 0 || component < 0 || component >= input_dim || size < 0 || size % input_dim != 0 || (d1_req_grad && !d1_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size / input_dim; i++){
        if(d1_req_grad) d1_grad[i * input_dim + component] += out_grad[i];
    }
    return 0;
}

int cpu_select_d2_backward(const float *out_grad, float *d2_grad, int input_dim, int p, int q, int size, int d2_req_grad){
    if(!out_grad || input_dim <= 0 || p < 0 || p >= input_dim || q < 0 || q >= input_dim || size < 0 || size % (input_dim * input_dim) != 0 || (d2_req_grad && !d2_grad)) return 1;

    int channels = input_dim * input_dim;
    #pragma omp parallel for
    for(int i = 0; i < size / channels; i++){
        if(d2_req_grad) d2_grad[i * channels + p * input_dim + q] += out_grad[i];
    }
    return 0;
}

int cpu_select_col_backward(const float *out_grad, float *a_grad, int component, int rows, int cols, int a_req_grad){
    if(!out_grad || rows < 0 || cols <= 0 || component < 0 || component >= cols || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < rows; i++){
        if(a_req_grad) a_grad[i * cols + component] += out_grad[i];
    }
    return 0;
}

int cpu_relu_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i] * (out[i] > 0.0f ? 1.0f : 0.0f);
    }
    return 0;
}

int cpu_sigmoid_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(a_req_grad) a_grad[i] += out_grad[i] * out[i] * (1.0f - out[i]);
    }
    return 0;
}
