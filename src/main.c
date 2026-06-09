/*
 * src/main.c
 *
 * Temporary executable for framework smoke tests. Use this to validate tensor
 * allocation, tensor ops, autograd, and small MLP tests before moving examples
 * into problem-specific directories such as Black-Scholes/1D.
 */

#include <stdio.h>
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"

int main(void) {
    int shape[1] = {3};
    float values[3] = {1.0f, 2.0f, 3.0f};

    Tensor *x = tensor_from_data(values, shape, 1, 1);

    tensor_print(x, "x");

    tensor_fill(x, 5.0f);
    tensor_print(x, "filled x");

    tensor_zero(x);
    tensor_print(x, "zeroed x");

    tensor_zero_grad(x);
    tensor_free(x);

    float a_values[3] = {1.0f, 2.0f, 3.0f};
    float b_values[3] = {10.0f, 20.0f, 30.0f};

    Tensor *a = tensor_from_data(a_values, shape, 1, 1);
    Tensor *b = tensor_from_data(b_values, shape, 1, 1);

    Tensor *sum = tensor_add(a, b);
    Tensor *prod = tensor_mult(a, b);

    tensor_print(sum, "a + b");
    tensor_print(prod, "a * b");

    Tensor *sq = tensor_square(a);
    Tensor *mean_sq = tensor_mean(sq);
    Tensor *mse = tensor_mse(a, b);

    tensor_print(sq, "a squared");
    tensor_print(mean_sq, "mean(a squared)");
    tensor_print(mse, "mse(a, b)");

    int mat_a_shape[2] = {2, 3};
    int mat_b_shape[2] = {3, 2};
    int bias_shape[1] = {2};

    float mat_a_values[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float mat_b_values[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    float bias_values[2] = {100.0f, 200.0f};

    Tensor *mat_a = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    Tensor *mat_b = tensor_from_data(mat_b_values, mat_b_shape, 2, 1);
    Tensor *bias = tensor_from_data(bias_values, bias_shape, 1, 1);

    Tensor *mat_prod = tensor_matmult(mat_a, mat_b);
    Tensor *biased = tensor_bias_add(mat_prod, bias);

    tensor_print(mat_prod, "mat_a @ mat_b");
    tensor_print(biased, "(mat_a @ mat_b) + bias");

    tensor_free(sq);
    tensor_free(mean_sq);
    tensor_free(mse);
    tensor_free(mat_prod);
    tensor_free(biased);
    tensor_free(mat_a);
    tensor_free(mat_b);
    tensor_free(bias);
    tensor_free(sum);
    tensor_free(prod);
    tensor_free(a);
    tensor_free(b);


    return 0;
}
