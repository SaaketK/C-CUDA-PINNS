#ifndef OPS_CUDA_H
#define OPS_CUDA_H

// CUDA Ops
#ifdef __cplusplus
extern "C" {
#endif

int cuda_add(const float *a, const float *b, float *out, int size);
int cuda_sub(const float *a, const float *b, float *out, int size);
int cuda_mult(const float *a, const float *b, float *out, int size);
int cuda_matmult(const float *a, const float *b, float *out, int rows, int inner, int cols);
int cuda_bias_add(const float *a, const float *b, float *out, int rows, int cols);
int cuda_mse(const float *a, const float *b, float *out, int size);
int cuda_mean(const float *a, float *out, int size);
int cuda_square(const float *a, float *out, int size);
int cuda_tanh(const float *a, float *out, int size);
int cuda_scalar_mult(const float *a, float scalar, float *out, int size);
int cuda_scalar_add(const float *a, float scalar, float *out, int size);
int cuda_identity(const float *a, float *out, int size);
int cuda_scale_deriv(const float *deriv, const float *factor, int input_dim, float *out, int size);
int cuda_chain_d2(const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, int size, float *out);
int cuda_deriv_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out);
int cuda_deriv2_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out);
int cuda_select_d1(const float *d1, int input_dim, int component, float *out, int size);
int cuda_select_d2(const float *d2, int input_dim, int p, int q, float *out, int size);
int cuda_select_col(const float *a, int component, float *out, int rows, int cols);
int cuda_relu(const float *a, float *out, int size);
int cuda_sigmoid(const float *a, float *out, int size);

#ifdef __cplusplus
}
#endif

#endif
