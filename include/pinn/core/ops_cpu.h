#ifndef OPS_CPU_H
#define OPS_CPU_H

int cpu_add(const float *a, const float *b, float *out, int size);
int cpu_sub(const float *a, const float *b, float *out, int size);
int cpu_mul(const float *a, const float *b, float *out, int size);
int cpu_matmult(const float *a, const float *b, float *out, int rows, int inner, int cols);
int cpu_bias_add(const float *a, const float *b, float *out, int rows, int cols);
int cpu_mse(const float *a, const float *b, float *out, int size);
int cpu_mean(const float *a, float *out, int size);
int cpu_square(const float *a, float *out, int size);
int cpu_tanh(const float *a, float *out, int size);
int cpu_scalar_mult(const float *a, float scalar, float *out, int size);
int cpu_scalar_add(const float *a, float scalar, float *out, int size);
int cpu_identity(const float *a, float *out, int size);
int cpu_scale_deriv(const float *deriv, const float *factor, int input_dim, float *out, int size);
int cpu_chain_d2(const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, int size, float *out);
int cpu_deriv_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out);
int cpu_deriv2_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out);
int cpu_select_d1(const float *d1, int input_dim, int component, float *out, int size);
int cpu_select_d2(const float *d2, int input_dim, int p, int q, float *out, int size);
int cpu_select_col(const float *a, int component, float *out, int rows, int cols);
int cpu_relu(const float *a, float *out, int size);
int cpu_sigmoid(const float *a, float *out, int size);

#endif
