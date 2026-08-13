#include "pinn/core/ops_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>

static cublasHandle_t g_cublas_handle = NULL;

static int get_cublas_handle(cublasHandle_t *handle){
    if(g_cublas_handle == NULL){
        if(cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS) return 1;
    }
    *handle = g_cublas_handle;
    return 0;
}

// Forward Operations

static __global__ void add_kernel(const float *a, const float *b, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] + b[i];
    }
}

extern "C" int cuda_add(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    add_kernel<<<blocks, threads_per_block>>>(a, b, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        return 1;
    }
    return 0;
}

static __global__ void sub_kernel(const float *a, const float *b, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] - b[i];
    }
}

extern "C" int cuda_sub(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    sub_kernel<<<blocks, threads_per_block>>>(a, b, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        return 1;
    }
    return 0;
}

static __global__ void mult_kernel(const float *a, const float *b, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] * b[i];
    }
}

extern "C" int cuda_mult(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    mult_kernel<<<blocks, threads_per_block>>>(a, b, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void mean_kernel(const float *a, float *out, int size){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        float sum = 0.0f;
        for(int i = 0; i < size; i++){
            sum += a[i];
        }
        out[0] = sum / size;
    }
}

extern "C" int cuda_mean(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    mean_kernel<<<1, 1>>>(a, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void square_kernel(const float *a, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] * a[i];
    }
}

extern "C" int cuda_square(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    square_kernel<<<blocks, threads_per_block>>>(a, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void tanh_kernel(const float *a, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = tanhf(a[i]);
    }
}

extern "C" int cuda_tanh(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    tanh_kernel<<<blocks, threads_per_block>>>(a, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void scalar_mult_kernel(const float *a, float scalar, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] * scalar;
    }
}

extern "C" int cuda_scalar_mult(const float *a, float scalar, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    scalar_mult_kernel<<<blocks, threads_per_block>>>(a, scalar, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void scalar_add_kernel(const float *a, float scalar, float *out, int size){
    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] + scalar;
    }
}

extern "C" int cuda_scalar_add(const float *a, float scalar, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    scalar_add_kernel<<<blocks, threads_per_block>>>(a, scalar, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void identity_kernel(const float *a, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i];
    }
}

extern "C" int cuda_identity(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    identity_kernel<<<blocks, threads_per_block>>>(a, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

extern "C" int cuda_matmult(const float *a, const float *b, float *out, int rows, int inner, int cols){
    if(!a || !b || !out || rows < 0 || inner < 0 || cols < 0) return 1;
    if(rows == 0 || inner == 0 || cols == 0) return 0;

    cublasHandle_t handle;
    if(get_cublas_handle(&handle) != 0) return 1;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
     * Tensor storage is row-major, while cuBLAS treats buffers as
     * column-major.  A row-major A (rows x inner) is A^T in cuBLAS,
     * and likewise for B.  Therefore compute:
     *
     *     C^T = B^T * A^T
     *
     * which writes the desired row-major C into the same buffer.
     */
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inner, &alpha, b, cols, a, inner, &beta, out, cols);

    if(status != CUBLAS_STATUS_SUCCESS) return 1;
    return 0;
}

static __global__ void bias_add_kernel(const float *a, const float *bias, float *out, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rows * cols){
        int col = i % cols;
        out[i] = a[i] + bias[col];
    }
}

extern "C" int cuda_bias_add(const float *a, const float *b, float *out, int rows, int cols){
    if(!a || !b || !out || rows < 0 || cols < 0) return 1;
    if(rows == 0 || cols == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;
    bias_add_kernel<<<blocks, threads_per_block>>>(a, b, out, rows, cols);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

extern "C" int cuda_mse(const float *a, const float *b, float *out, int size){
    if(!a || !b || !out || size < 0) return 1;
    if(size == 0) return 0;

    if(cuda_sub(a, b, out, size) != 0) return 1;
    if(cuda_square(out, out, size) != 0) return 1;
    if(cuda_mean(out, out, size) != 0) return 1;
    return 0;
}

static __global__ void scale_deriv_kernel(const float *deriv, const float *factor, int input_dim, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = deriv[i] * factor[i / input_dim];
    }
}

extern "C" int cuda_scale_deriv(const float *deriv, const float* factor, int input_dim, float *out, int size){
    if(!deriv || !factor || input_dim <= 0 || !out || size < 0 || size % input_dim != 0) return 1;
    if(size == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    scale_deriv_kernel<<<blocks, threads_per_block>>>(deriv, factor, input_dim, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void chain_d2_kernel(const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, int size, float *out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channels = input_dim * input_dim;
    int total = size * channels;

    if(idx < total){
        int value = idx / channels;
        int rem = idx % channels;
        int p = rem / input_dim;
        int q = rem % input_dim;
        int d1_p = value * input_dim + p;
        int d1_q = value * input_dim + q;

        out[idx] = f_double[value] * d1[d1_p] * d1[d1_q] + f_prime[value] * d2[idx];
    }
}

extern "C" int cuda_chain_d2(const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, int size, float *out){
    if(!d1 || !d2 || !f_prime || !f_double || !out || input_dim <= 0 || size < 0) return 1;
    if(size == 0) return 0;

    const int channels = input_dim * input_dim;
    const int total = size * channels;
    const int threads_per_block = 256;
    const int blocks = (total + threads_per_block - 1) / threads_per_block;

    chain_d2_kernel<<<blocks, threads_per_block>>>(d1, d2, f_prime, f_double, input_dim, size, out);

    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? 0 : 1;
}

static __global__ void deriv_matmult_kernel(const float *deriv, const float *W, float *out, int batch, int in_features, int out_features, int channels){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features * channels;

    if(idx < total){
        int output_value = idx / channels;
        int ch = idx % channels;
        int r = output_value / out_features;
        int c = output_value % out_features;

        float sum = 0.0f;
        for(int k = 0; k < in_features; ++k){
            int deriv_idx = (r * in_features + k) * channels + ch;
            int weight_idx = k * out_features + c;
            sum += deriv[deriv_idx] * W[weight_idx];
        }

        out[idx] = sum;
    }
}

static int cuda_deriv_matmult_order(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, int order, float *out){
    if(!deriv || !W || !out || batch < 0 || in_features < 0 || out_features < 0 || input_dim <= 0) return 1;
    if(batch == 0 || in_features == 0 || out_features == 0) return 0;

    int channels = 1;
    for(int i = 0; i < order; ++i){
        channels *= input_dim;
    }

    const int total = batch * out_features * channels;
    const int threads_per_block = 256;
    const int blocks = (total + threads_per_block - 1) / threads_per_block;

    deriv_matmult_kernel<<<blocks, threads_per_block>>>(deriv, W, out, batch, in_features, out_features, channels);

    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? 0 : 1;
}

extern "C" int cuda_deriv_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out){
    return cuda_deriv_matmult_order(deriv, W, batch, in_features, out_features, input_dim, 1, out);
}

extern "C" int cuda_deriv2_matmult(const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out){
    return cuda_deriv_matmult_order(deriv, W, batch, in_features, out_features, input_dim, 2, out);
}

static __global__ void select_d1_kernel(const float *d1, int input_dim, int component, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = size / input_dim;
    if(i < out_size){
        out[i] = d1[i * input_dim + component];
    }
}

extern "C" int cuda_select_d1(const float *d1, int input_dim, int component, float *out, int size){
    if(!d1 || !out || input_dim <= 0 || component < 0 || component >= input_dim || size < 0 || size % input_dim != 0) return 1;
    if(size == 0) return 0;

    const int out_size = size / input_dim;
    const int threads_per_block = 256;
    const int blocks = (out_size + threads_per_block - 1) / threads_per_block;
    select_d1_kernel<<<blocks, threads_per_block>>>(d1, input_dim, component, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void select_d2_kernel(const float *d2, int input_dim, int p, int q, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int channels = input_dim * input_dim;
    int out_size = size / channels;
    if(i < out_size){
        out[i] = d2[i * channels + p * input_dim + q];
    }
}

extern "C" int cuda_select_d2(const float *d2, int input_dim, int p, int q, float *out, int size){
    if(!d2 || !out || input_dim <= 0 || p < 0 || p >= input_dim || q < 0 || q >= input_dim || size < 0 || size % (input_dim * input_dim) != 0) return 1;
    if(size == 0) return 0;

    const int channels = input_dim * input_dim;
    const int out_size = size / channels;
    const int threads_per_block = 256;
    const int blocks = (out_size + threads_per_block - 1) / threads_per_block;
    select_d2_kernel<<<blocks, threads_per_block>>>(d2, input_dim, p, q, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void select_col_kernel(const float *a, int component, float *out, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rows){
        out[i] = a[i * cols + component];
    }
}

extern "C" int cuda_select_col(const float *a, int component, float *out, int rows, int cols){
    if(!a || !out || rows < 0 || cols <= 0 || component < 0 || component >= cols) return 1;
    if(rows == 0) return 0;

    const int threads_per_block = 256;
    const int blocks = (rows + threads_per_block - 1) / threads_per_block;
    select_col_kernel<<<blocks, threads_per_block>>>(a, component, out, rows, cols);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void relu_kernel(const float *a, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a[i] > 0){
            out[i] = a[i];
        }
        else {
            out[i] = 0;
        }
    }
}

extern "C" int cuda_relu(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(a, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void sigmoid_kernel(const float *a, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = 1.0f / (1.0f + exp(-1 * a[i]));
    }
}

extern "C" int cuda_sigmoid(const float *a, float *out, int size){
    if(!a || !out || size < 0) return 1;
    if(size == 0) return 0;

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<blocks, threads_per_block>>>(a, out, size);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

// Backward Operations

static __global__ void add_backward_kernel(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad) a_grad[i] += out_grad[i];
        if(b_req_grad) b_grad[i] += out_grad[i];
    }
}

extern "C" int cuda_add_backward(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;
    if(size == 0 || (!a_req_grad && !b_req_grad)) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    add_backward_kernel<<<blocks, threads_per_block>>>(out_grad, a_grad, b_grad, size, a_req_grad, b_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void sub_backward_kernel(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad) a_grad[i] += out_grad[i];
        if(b_req_grad) b_grad[i] -= out_grad[i];
    }
}

extern "C" int cuda_sub_backward(const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;
    if(size == 0 || (!a_req_grad && !b_req_grad)) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    sub_backward_kernel<<<blocks, threads_per_block>>>(out_grad, a_grad, b_grad, size, a_req_grad, b_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void mult_backward_kernel(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad) a_grad[i] += out_grad[i] * b[i];
        if(b_req_grad) b_grad[i] += out_grad[i] * a[i];
    }
}

extern "C" int cuda_mult_backward(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int size, int a_req_grad, int b_req_grad){
    if(!a || !b || !out_grad || size < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;
    if(size == 0 || (!a_req_grad && !b_req_grad)) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    mult_backward_kernel<<<blocks, threads_per_block>>>(a, b, out_grad, a_grad, b_grad, size, a_req_grad, b_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

extern "C" int cuda_matmult_backward(const float *a, const float *b, const float *out_grad, float *a_grad, float *b_grad, int rows, int inner, int cols, int a_req_grad, int b_req_grad){
    if(!a || !b || !out_grad || rows < 0 || inner < 0 || cols < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;
    if(rows == 0 || inner == 0 || cols == 0 || (!a_req_grad && !b_req_grad)) return 0;

    cublasHandle_t handle;
    if(get_cublas_handle(&handle) != 0) return 1;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    if(a_req_grad){
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, inner, rows, cols, &alpha, b, cols, out_grad, cols, &beta, a_grad, inner);
        if(status != CUBLAS_STATUS_SUCCESS) return 1;
    }
    if(b_req_grad){
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, cols, inner, rows, &alpha, out_grad, cols, a, inner, &beta, b_grad, cols);
        if(status != CUBLAS_STATUS_SUCCESS) return 1;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void bias_add_a_backward_kernel(const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size && a_req_grad) a_grad[i] += out_grad[i];
}

static __global__ void bias_add_b_backward_kernel(const float *out_grad, float *b_grad, int rows, int cols, int b_req_grad){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < cols && b_req_grad){
        float sum = 0.0f;
        for(int row = 0; row < rows; row++) sum += out_grad[row * cols + col];
        b_grad[col] += sum;
    }
}

extern "C" int cuda_bias_add_backward(const float *out_grad, float *a_grad, float *b_grad, int rows, int cols, int a_req_grad, int b_req_grad){
    if(!out_grad || rows < 0 || cols < 0 || (a_req_grad && !a_grad) || (b_req_grad && !b_grad)) return 1;
    if(rows == 0 || cols == 0 || (!a_req_grad && !b_req_grad)) return 0;

    const int threads_per_block = 256;
    if(a_req_grad){
        const int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;
        bias_add_a_backward_kernel<<<blocks, threads_per_block>>>(out_grad, a_grad, rows * cols, a_req_grad);
    }
    if(b_req_grad){
        const int blocks = (cols + threads_per_block - 1) / threads_per_block;
        bias_add_b_backward_kernel<<<blocks, threads_per_block>>>(out_grad, b_grad, rows, cols, b_req_grad);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void mean_backward_kernel(const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad){
            a_grad[i] += out_grad[0] / size;
        }
    }
}

extern "C" int cuda_mean_backward(const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    mean_backward_kernel<<<blocks, threads_per_block>>>(out_grad, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void square_backward_kernel(const float *a, const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad){
            a_grad[i] += 2.0f * out_grad[i] * a[i];
        }
    }
}

extern "C" int cuda_square_backward(const float *a, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!a || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    square_backward_kernel<<<blocks, threads_per_block>>>(a, out_grad, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void tanh_backward_kernel(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad){
            a_grad[i] += out_grad[i] * (1.0f - out[i] * out[i]);
        }
    }
}

extern "C" int cuda_tanh_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    tanh_backward_kernel<<<blocks, threads_per_block>>>(out, out_grad, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

extern "C" int cuda_identity_backward(const float *out_grad, float *a_grad, int size, int a_req_grad){
    return cuda_scalar_add_backward(out_grad, a_grad, size, a_req_grad);
}

static __global__ void scalar_mult_backward_kernel(const float *out_grad, float scalar, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(a_req_grad){
            a_grad[i] += out_grad[i] * scalar;
        }
    }
}

extern "C" int cuda_scalar_mult_backward(const float *out_grad, float scalar, float *a_grad, int size, int a_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    scalar_mult_backward_kernel<<<blocks, threads_per_block>>>(out_grad, scalar, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void scalar_add_backward_kernel(const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size && a_req_grad){
        a_grad[i] += out_grad[i];
    }
}

extern "C" int cuda_scalar_add_backward(const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    scalar_add_backward_kernel<<<blocks, threads_per_block>>>(out_grad, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void scale_deriv_backward_kernel(const float *deriv, const float* factor, const float *out_grad, float *deriv_grad, float *factor_grad, int input_dim, int size,
int deriv_req_grad, int factor_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        if(deriv_req_grad){
            deriv_grad[i] += out_grad[i] * factor[i / input_dim];
        }
        if(factor_req_grad){
            atomicAdd(&factor_grad[i / input_dim], out_grad[i] * deriv[i]);
        }
    }
}

extern "C" int cuda_scale_deriv_backward(const float *deriv, const float *factor, const float *out_grad, float *deriv_grad, float *factor_grad, int input_dim, int size,
int deriv_req_grad, int factor_req_grad){
    if(!deriv || !factor || !out_grad || input_dim <= 0 || size < 0 || size % input_dim != 0 || (deriv_req_grad && !deriv_grad) || (factor_req_grad && !factor_grad)) return 1;
    if(size == 0 || (!deriv_req_grad && !factor_req_grad)) return 0;

    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    scale_deriv_backward_kernel<<<blocks, threads_per_block>>>(deriv, factor, out_grad, deriv_grad, factor_grad, input_dim, size, deriv_req_grad, factor_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}


static __global__ void chain_d2_backward_kernel(const float *d1, const float *d2, const float *f_prime, const float *f_double, const float *out_grad,
float *d1_grad, float *d2_grad, float *f_prime_grad, float *f_double_grad, int input_dim, int size, int d1_req_grad, int d2_req_grad, int f_prime_req_grad, int f_double_req_grad){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channels = input_dim * input_dim;
    int total = size * channels;

    if(idx < total){
        int value = idx / channels;
        int rem = idx % channels;
        int p = rem / input_dim;
        int q = rem % input_dim;
        int idxp = value * input_dim + p;
        int idxq = value * input_dim + q;

        float grad = out_grad[idx];

        if(d1_req_grad){
            atomicAdd(&d1_grad[idxp], grad * f_double[value] * d1[idxq]);
            atomicAdd(&d1_grad[idxq], grad * f_double[value] * d1[idxp]);
        }
        if(d2_req_grad){
            d2_grad[idx] += grad * f_prime[value];
        }
        if(f_prime_req_grad){
            atomicAdd(&f_prime_grad[value], grad * d2[idx]);
        }
        if(f_double_req_grad){
            atomicAdd(&f_double_grad[value], grad * d1[idxp] * d1[idxq]);
        }
    }
}

extern "C" int cuda_chain_d2_backward(const float *d1, const float *d2, const float *f_prime, const float *f_double, const float *out_grad,
float *d1_grad, float *d2_grad, float *f_prime_grad, float *f_double_grad, int input_dim, int size, int d1_req_grad, int d2_req_grad, int f_prime_req_grad, int f_double_req_grad){
    if(!d1 || !d2 || !f_prime || !f_double || !out_grad || input_dim <= 0 || size < 0 || (d1_req_grad && !d1_grad) || (d2_req_grad && !d2_grad) || (f_prime_req_grad && !f_prime_grad) || (f_double_req_grad && !f_double_grad)) return 1;
    if(size == 0 || (!d1_req_grad && !d2_req_grad && !f_prime_req_grad && !f_double_req_grad)) return 0;

    int channels = input_dim * input_dim;
    int total = size * channels;
    const int threads_per_block = 256;
    const int blocks = (total + threads_per_block - 1) / threads_per_block;
    chain_d2_backward_kernel<<<blocks, threads_per_block>>>(d1, d2, f_prime, f_double, out_grad, d1_grad, d2_grad, f_prime_grad, f_double_grad, input_dim, size,
        d1_req_grad, d2_req_grad, f_prime_req_grad, f_double_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

extern "C" int cuda_deriv_matmult_backward(const float *deriv, const float *W, const float *out_grad, float *deriv_grad, float *W_grad, int batch, int in_features, int out_features,
int input_dim, int order, int deriv_req_grad, int W_req_grad){
    if(!deriv || !W || !out_grad || batch < 0 || in_features < 0 || out_features < 0 || input_dim <= 0 || order <= 0 || (deriv_req_grad && !deriv_grad) || (W_req_grad && !W_grad)) return 1;
    if(batch == 0 || in_features == 0 || out_features == 0 || (!deriv_req_grad && !W_req_grad)) return 0;

    int channels = 1;
    for(int i = 0; i < order; i++) channels *= input_dim;

    cublasHandle_t handle;
    if(get_cublas_handle(&handle) != 0) return 1;

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const long long deriv_stride = (long long)in_features * channels;
    const long long out_stride = (long long)out_features * channels;

    if(deriv_req_grad){
        cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, channels, in_features, out_features, &alpha, out_grad, channels, out_stride, W, out_features, 0, &beta, deriv_grad, channels, deriv_stride, batch);
        if(status != CUBLAS_STATUS_SUCCESS) return 1;
    }

    if(W_req_grad){
        for(int r = 0; r < batch; r++){
            const float *out_grad_batch = out_grad + r * out_features * channels;
            const float *deriv_batch = deriv + r * in_features * channels;
            cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, in_features, channels, &alpha, out_grad_batch, channels, deriv_batch, channels, &beta, W_grad, out_features);
            if(status != CUBLAS_STATUS_SUCCESS) return 1;
        }
    }

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void select_d1_backward_kernel(const float *out_grad, float *d1_grad, int input_dim, int component, int size, int d1_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_dim = size / input_dim;

    if(i < out_dim && d1_req_grad){
        d1_grad[i * input_dim + component] += out_grad[i];
    }
}

extern "C" int cuda_select_d1_backward(const float *out_grad, float *d1_grad, int input_dim, int component, int size, int d1_req_grad){
    if(!out_grad || input_dim <= 0 || component < 0 || component >= input_dim || size < 0 || size % input_dim != 0 || (d1_req_grad && !d1_grad)) return 1;
    if(size == 0 || !d1_req_grad) return 0;

    int out_dim = size / input_dim;
    int threads_per_block = 256;
    int blocks = (out_dim + threads_per_block - 1) / threads_per_block;
    select_d1_backward_kernel<<<blocks, threads_per_block>>>(out_grad, d1_grad, input_dim, component, size, d1_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void select_d2_backward_kernel(const float *out_grad, float *d2_grad, int input_dim, int p, int q, int size, int d2_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_dim = size / (input_dim * input_dim);

    if(i < out_dim && d2_req_grad){
        d2_grad[i * input_dim * input_dim + p * input_dim + q] += out_grad[i];
    }
}

extern "C" int cuda_select_d2_backward(const float *out_grad, float *d2_grad, int input_dim, int p, int q, int size, int d2_req_grad){
    if(!out_grad || input_dim <= 0 || p < 0 || p >= input_dim || q < 0 || q >= input_dim || size < 0 || size % (input_dim * input_dim) != 0 || (d2_req_grad && !d2_grad)) return 1;
    if(size == 0 || !d2_req_grad) return 0;

    int out_dim = size / (input_dim * input_dim);
    int threads_per_block = 256;
    int blocks = (out_dim + threads_per_block - 1) / threads_per_block;
    select_d2_backward_kernel<<<blocks, threads_per_block>>>(out_grad, d2_grad, input_dim, p, q, size, d2_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void select_col_backward_kernel(const float *out_grad, float *a_grad, int component, int rows, int cols, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < rows && a_req_grad){
        a_grad[i * cols + component] += out_grad[i];
    }
}

extern "C" int cuda_select_col_backward(const float *out_grad, float *a_grad, int component, int rows, int cols, int a_req_grad){
    if(!out_grad || rows < 0 || cols <= 0 || component < 0 || component >= cols || (a_req_grad && !a_grad)) return 1;
    if(rows == 0 || !a_req_grad) return 0;

    int threads_per_block = 256;
    int blocks = (rows + threads_per_block - 1) / threads_per_block;
    select_col_backward_kernel<<<blocks, threads_per_block>>>(out_grad, a_grad, component, rows, cols, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void relu_backward_kernel(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size && a_req_grad){
        a_grad[i] += out_grad[i] * (out[i] > 0 ? 1.0f : 0.0f);
    }
}

extern "C" int cuda_relu_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    relu_backward_kernel<<<blocks, threads_per_block>>>(out, out_grad, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}

static __global__ void sigmoid_backward_kernel(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size && a_req_grad){
        a_grad[i] += out_grad[i] * out[i] * (1.0f - out[i]);
    }
}

extern "C" int cuda_sigmoid_backward(const float *out, const float *out_grad, float *a_grad, int size, int a_req_grad){
    if(!out || !out_grad || size < 0 || (a_req_grad && !a_grad)) return 1;
    if(size == 0 || !a_req_grad) return 0;

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    sigmoid_backward_kernel<<<blocks, threads_per_block>>>(out, out_grad, a_grad, size, a_req_grad);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) return 1;
    return 0;
}
