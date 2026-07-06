#ifndef OPS_CUDA_H
#define OPS_CUDA_H

// CUDA Ops
void cuda_add(const float *a, const float *b, float *out);
void cuda_sub(const float *a, const float *b, float *out);
void cuda_mult(const float *a, const float *b, float *out);
void cuda_matmult(const float *a, const float *b, float *out);
void cuda_bias_add(const float *a, const float *b, float *out);
void cuda_mse(const float *a, const float *b, float *out);
void cuda_mean(const float *a, float *out);
void cuda_square(const float *a, float *out);
void cuda_tanh(const float *a, float *out);
void cuda_scalar_mult(const float *a, float scalar, float *out);
void cuda_scalar_add(const float *a, float scalar, float *out);
void cuda_identity(const float *a, float *out);
void cuda_scale_deriv(const float *deriv, const float *factor, int input_dim, float *out);
void cuda_chain_d2(const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, float *out);
void cuda_deriv_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out);
void cuda_deriv2_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out);
void cuda_select_d1(const float *d1, int input_dim, int component, float *out);
void cuda_select_d2(const float *d2, int input_dim, int p, int q, float *out);
void cuda_select_col(const float *a, int component, float *out);
void cuda_relu(const float *a, float *out);
void cuda_sigmoid(const float *a, float *out);

#endif