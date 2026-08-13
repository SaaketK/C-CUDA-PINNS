#ifndef OPS_CUDA_H
#define OPS_CUDA_H

// CUDA Ops
#ifdef __cplusplus
extern "C" {
#endif

// Forward ops

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

// Backward ops

int cuda_add_backward(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad);
int cuda_sub_backward(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad);
int cuda_mult_backward(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad);
int cuda_matmult_backward(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int rows, int inner, int cols, int a_req_grad, int b_req_grad);
int cuda_bias_add_backward(const float *out_grad, float *a_grad, float *b_grad, int rows, int cols, int a_req_grad, int b_req_grad);
int cuda_mean_backward(const float *out_grad, float *a_grad, int size, int a_req_grad);
int cuda_square_backward(const float *a, const float *out_grad, float *a_grad, int size, int a_req_grad);
int cuda_tanh_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad);
int cuda_scalar_mult_backward(const float *out_grad, float scalar, float *a_grad, int size, int a_req_grad);
int cuda_scalar_add_backward(const float *out_grad, float *a_grad, int size, int a_req_grad);
int cuda_identity_backward(const float *out_grad, float *a_grad, int size, int a_req_grad);
int cuda_scale_deriv_backward(const float *deriv, const float *factor, const float *out_grad, float *deriv_grad, float *factor_grad, int input_dim, int size, int deriv_req_grad, int factor_req_grad);
int cuda_chain_d2_backward(const float *d1, const float *d2, const float *f_prime, const float *f_double, const float *out_grad, float *d1_grad, float *d2_grad, float *f_prime_grad, float *f_double_grad, int input_dim, int size, int d1_req_grad, int d2_req_grad, int f_prime_req_grad, int f_double_req_grad);
int cuda_deriv_matmult_backward(const float *deriv, const float *W, const float *out_grad, float *deriv_grad, float *W_grad, int batch, int in_features, int out_features, int input_dim, int order, int deriv_req_grad, int W_req_grad);
int cuda_select_d1_backward(const float *out_grad, float *d1_grad, int input_dim, int component, int size, int d1_req_grad);
int cuda_select_d2_backward(const float *out_grad, float *d2_grad, int input_dim, int p, int q, int size, int d2_req_grad);
int cuda_select_col_backward(const float *out_grad, float *a_grad, int component, int rows, int cols, int a_req_grad);
int cuda_relu_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad);
int cuda_sigmoid_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad);


#ifdef __cplusplus
}
#endif

#endif
