#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pinn/core/backend.h"
#include "pinn/core/ops_cpu.h"

#ifdef PINN_USE_CUDA
#include "pinn/core/ops_cuda.h"
#endif

typedef int (*binary_op)(const float *, const float *, float *, int);
typedef int (*unary_op)(const float *, float *, int);
typedef int (*scalar_op)(const float *, float, float *, int);
typedef int (*matmult_op)(const float *, const float *, float *, int, int, int);
typedef int (*bias_op)(const float *, const float *, float *, int, int);
typedef int (*scale_op)(const float *, const float *, int, float *, int);
typedef int (*chain_op)(const float *, const float *, const float *, const float *, int, int, float *);
typedef int (*deriv_op)(const float *, const float *, int, int, int, int, float *);
typedef int (*select_d1_op)(const float *, int, int, float *, int);
typedef int (*select_d2_op)(const float *, int, int, int, float *, int);
typedef int (*select_col_op)(const float *, int, float *, int, int);

static double elapsed_seconds(struct timespec start, struct timespec end){
    return (double)(end.tv_sec - start.tv_sec) + 1e-9 * (double)(end.tv_nsec - start.tv_nsec);
}

static double bench_binary(binary_op op, const float *a, const float *b, float *out, int size, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, b, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_unary(unary_op op, const float *a, float *out, int size, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_scalar(scalar_op op, const float *a, float scalar, float *out, int size, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, scalar, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_matmult(matmult_op op, const float *a, const float *b, float *out, int rows, int inner, int cols, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, b, out, rows, inner, cols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_bias(bias_op op, const float *a, const float *b, float *out, int rows, int cols, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, b, out, rows, cols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_scale(scale_op op, const float *deriv, const float *factor, int input_dim, float *out, int size, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(deriv, factor, input_dim, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_chain(chain_op op, const float *d1, const float *d2, const float *f_prime, const float *f_double, int input_dim, int size, float *out, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(d1, d2, f_prime, f_double, input_dim, size, out);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_deriv(deriv_op op, const float *deriv, const float *W, int batch, int in_features, int out_features, int input_dim, float *out, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(deriv, W, batch, in_features, out_features, input_dim, out);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_select_d1(select_d1_op op, const float *a, int input_dim, int component, float *out, int size, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, input_dim, component, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_select_d2(select_d2_op op, const float *a, int input_dim, int p, int q, float *out, int size, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, input_dim, p, q, out, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

static double bench_select_col(select_col_op op, const float *a, int component, float *out, int rows, int cols, int repeats){
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < repeats; i++) op(a, component, out, rows, cols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_seconds(start, end) * 1000.0 / repeats;
}

typedef struct {
    const char *name;
    double cpu_ms;
    double cuda_ms;
    int has_cuda;
} Timing;

static Timing timings[32];
static int timing_count = 0;

static int timing_index(const char *name){
    for(int i = 0; i < timing_count; i++){
        if(strcmp(timings[i].name, name) == 0) return i;
    }
    timings[timing_count].name = name;
    return timing_count++;
}

static void record_cpu(const char *name, double ms){
    int index = timing_index(name);
    timings[index].cpu_ms = ms;
}

static void record_cuda(const char *name, double ms){
    int index = timing_index(name);
    timings[index].cuda_ms = ms;
    timings[index].has_cuda = 1;
}

static void print_results(int has_cuda){
    printf("Operation                  CPU ms   CUDA ms    speedup\n");
    printf("--------------------------------------------------------\n");
    for(int i = 0; i < timing_count; i++){
        if(has_cuda && timings[i].has_cuda){
            printf("%-18s %10.6f %10.6f %10.3fx\n", timings[i].name, timings[i].cpu_ms, timings[i].cuda_ms, timings[i].cpu_ms / timings[i].cuda_ms);
        }
        else {
            printf("%-18s %10.6f %10s %10s\n", timings[i].name, timings[i].cpu_ms, "skip", "skip");
        }
    }
}

int main(void){
    const int size = 1 << 20;
    const int rows = 256;
    const int inner = 256;
    const int cols = 256;
    const int repeats = 20;
    const int matrix_repeats = 10;
    const int batch = 64;
    const int deriv_in = 64;
    const int deriv_out = 64;
    const int input_dim = 4;
    const int scale_size = 1 << 16;
    const int chain_size = 1 << 12;
    const int select_rows = 1 << 12;
    const int select_d1_size = select_rows * 4 * input_dim;
    const int select_d2_size = select_rows * 2 * input_dim * input_dim;

    float *a = (float*)malloc(size * sizeof(float));
    float *b = (float*)malloc(size * sizeof(float));
    float *out = (float*)malloc(size * sizeof(float));
    float *mat_a = (float*)malloc(rows * inner * sizeof(float));
    float *mat_b = (float*)malloc(inner * cols * sizeof(float));
    float *mat_out = (float*)malloc(rows * cols * sizeof(float));
    float *bias = (float*)malloc(cols * sizeof(float));
    float *scale_deriv = (float*)malloc(scale_size * sizeof(float));
    float *scale_factor = (float*)malloc((scale_size / input_dim) * sizeof(float));
    float *scale_out = (float*)malloc(scale_size * sizeof(float));
    float *chain_d1 = (float*)malloc(chain_size * input_dim * sizeof(float));
    float *chain_d2 = (float*)malloc(chain_size * input_dim * input_dim * sizeof(float));
    float *chain_f = (float*)malloc(chain_size * sizeof(float));
    float *chain_fd = (float*)malloc(chain_size * sizeof(float));
    float *chain_out = (float*)malloc(chain_size * input_dim * input_dim * sizeof(float));
    float *deriv1 = (float*)malloc(batch * deriv_in * input_dim * sizeof(float));
    float *deriv2 = (float*)malloc(batch * deriv_in * input_dim * input_dim * sizeof(float));
    float *W = (float*)malloc(deriv_in * deriv_out * sizeof(float));
    float *deriv1_out = (float*)malloc(batch * deriv_out * input_dim * sizeof(float));
    float *deriv2_out = (float*)malloc(batch * deriv_out * input_dim * input_dim * sizeof(float));
    float *select_d1 = (float*)malloc(select_d1_size * sizeof(float));
    float *select_d2 = (float*)malloc(select_d2_size * sizeof(float));
    float *select_out = (float*)malloc(select_d2_size * sizeof(float));
    float *select_col = (float*)malloc(select_rows * 16 * sizeof(float));
    float *col_out = (float*)malloc(select_rows * sizeof(float));

    if(!a || !b || !out || !mat_a || !mat_b || !mat_out || !bias || !scale_deriv || !scale_factor || !scale_out || !chain_d1 || !chain_d2 || !chain_f || !chain_fd || !chain_out || !deriv1 || !deriv2 || !W || !deriv1_out || !deriv2_out || !select_d1 || !select_d2 || !select_out || !select_col || !col_out){
        printf("benchmark allocation failed\n");
        return 1;
    }

    for(int i = 0; i < size; i++){
        a[i] = (float)(i % 17) * 0.01f;
        b[i] = (float)(i % 13) * 0.02f;
    }
    for(int i = 0; i < rows * inner; i++) mat_a[i] = (float)(i % 17) * 0.01f;
    for(int i = 0; i < inner * cols; i++) mat_b[i] = (float)(i % 13) * 0.02f;
    for(int i = 0; i < cols; i++) bias[i] = 0.1f;
    for(int i = 0; i < scale_size; i++) scale_deriv[i] = 0.01f;
    for(int i = 0; i < scale_size / input_dim; i++) scale_factor[i] = 0.5f;
    for(int i = 0; i < chain_size * input_dim; i++) chain_d1[i] = 0.01f;
    for(int i = 0; i < chain_size * input_dim * input_dim; i++) chain_d2[i] = 0.01f;
    for(int i = 0; i < chain_size; i++){ chain_f[i] = 0.5f; chain_fd[i] = 0.25f; }
    for(int i = 0; i < batch * deriv_in * input_dim; i++) deriv1[i] = 0.01f;
    for(int i = 0; i < batch * deriv_in * input_dim * input_dim; i++) deriv2[i] = 0.01f;
    for(int i = 0; i < deriv_in * deriv_out; i++) W[i] = 0.01f;
    for(int i = 0; i < select_d1_size; i++) select_d1[i] = 0.01f;
    for(int i = 0; i < select_d2_size; i++) select_d2[i] = 0.01f;
    for(int i = 0; i < select_rows * 16; i++) select_col[i] = 0.01f;

    record_cpu("add", bench_binary(cpu_add, a, b, out, size, repeats));

#ifdef PINN_USE_CUDA
    int has_cuda = backend_cuda_available();
#else
    int has_cuda = 0;
#endif

    record_cpu("sub", bench_binary(cpu_sub, a, b, out, size, repeats));
    record_cpu("mul", bench_binary(cpu_mul, a, b, out, size, repeats));
    record_cpu("mse", bench_binary(cpu_mse, a, b, out, size, repeats));
    record_cpu("mean", bench_unary(cpu_mean, a, out, size, repeats));
    record_cpu("square", bench_unary(cpu_square, a, out, size, repeats));
    record_cpu("tanh", bench_unary(cpu_tanh, a, out, size, repeats));
    record_cpu("scalar_mult", bench_scalar(cpu_scalar_mult, a, 2.0f, out, size, repeats));
    record_cpu("scalar_add", bench_scalar(cpu_scalar_add, a, 2.0f, out, size, repeats));
    record_cpu("identity", bench_unary(cpu_identity, a, out, size, repeats));
    record_cpu("matmult", bench_matmult(cpu_matmult, mat_a, mat_b, mat_out, rows, inner, cols, matrix_repeats));
    record_cpu("bias_add", bench_bias(cpu_bias_add, mat_a, bias, mat_out, rows, cols, repeats));
    record_cpu("scale_deriv", bench_scale(cpu_scale_deriv, scale_deriv, scale_factor, input_dim, scale_out, scale_size, repeats));
    record_cpu("chain_d2", bench_chain(cpu_chain_d2, chain_d1, chain_d2, chain_f, chain_fd, input_dim, chain_size, chain_out, repeats));
    record_cpu("deriv_matmult", bench_deriv(cpu_deriv_matmult, deriv1, W, batch, deriv_in, deriv_out, input_dim, deriv1_out, matrix_repeats));
    record_cpu("deriv2_matmult", bench_deriv(cpu_deriv2_matmult, deriv2, W, batch, deriv_in, deriv_out, input_dim, deriv2_out, matrix_repeats));
    record_cpu("select_d1", bench_select_d1(cpu_select_d1, select_d1, input_dim, 1, select_out, select_d1_size, repeats));
    record_cpu("select_d2", bench_select_d2(cpu_select_d2, select_d2, input_dim, 1, 1, select_out, select_d2_size, repeats));
    record_cpu("select_col", bench_select_col(cpu_select_col, select_col, 1, col_out, select_rows, 16, repeats));
    record_cpu("relu", bench_unary(cpu_relu, a, out, size, repeats));
    record_cpu("sigmoid", bench_unary(cpu_sigmoid, a, out, size, repeats));

#ifdef PINN_USE_CUDA
    if(has_cuda){
        float *d_a = cuda_malloc(size);
        float *d_b = cuda_malloc(size);
        float *d_out = cuda_malloc(size);
        float *d_mat_a = cuda_malloc(rows * inner);
        float *d_mat_b = cuda_malloc(inner * cols);
        float *d_mat_out = cuda_malloc(rows * cols);
        float *d_bias = cuda_malloc(cols);
        float *d_scale_deriv = cuda_malloc(scale_size);
        float *d_scale_factor = cuda_malloc(scale_size / input_dim);
        float *d_scale_out = cuda_malloc(scale_size);
        float *d_chain_d1 = cuda_malloc(chain_size * input_dim);
        float *d_chain_d2 = cuda_malloc(chain_size * input_dim * input_dim);
        float *d_chain_f = cuda_malloc(chain_size);
        float *d_chain_fd = cuda_malloc(chain_size);
        float *d_chain_out = cuda_malloc(chain_size * input_dim * input_dim);
        float *d_deriv1 = cuda_malloc(batch * deriv_in * input_dim);
        float *d_deriv2 = cuda_malloc(batch * deriv_in * input_dim * input_dim);
        float *d_W = cuda_malloc(deriv_in * deriv_out);
        float *d_deriv1_out = cuda_malloc(batch * deriv_out * input_dim);
        float *d_deriv2_out = cuda_malloc(batch * deriv_out * input_dim * input_dim);
        float *d_select_d1 = cuda_malloc(select_d1_size);
        float *d_select_d2 = cuda_malloc(select_d2_size);
        float *d_select_out = cuda_malloc(select_d2_size);
        float *d_select_col = cuda_malloc(select_rows * 16);
        float *d_col_out = cuda_malloc(select_rows);

        cuda_memcpy_to_device(a, d_a, size);
        cuda_memcpy_to_device(b, d_b, size);
        cuda_memcpy_to_device(mat_a, d_mat_a, rows * inner);
        cuda_memcpy_to_device(mat_b, d_mat_b, inner * cols);
        cuda_memcpy_to_device(bias, d_bias, cols);
        cuda_memcpy_to_device(scale_deriv, d_scale_deriv, scale_size);
        cuda_memcpy_to_device(scale_factor, d_scale_factor, scale_size / input_dim);
        cuda_memcpy_to_device(chain_d1, d_chain_d1, chain_size * input_dim);
        cuda_memcpy_to_device(chain_d2, d_chain_d2, chain_size * input_dim * input_dim);
        cuda_memcpy_to_device(chain_f, d_chain_f, chain_size);
        cuda_memcpy_to_device(chain_fd, d_chain_fd, chain_size);
        cuda_memcpy_to_device(deriv1, d_deriv1, batch * deriv_in * input_dim);
        cuda_memcpy_to_device(deriv2, d_deriv2, batch * deriv_in * input_dim * input_dim);
        cuda_memcpy_to_device(W, d_W, deriv_in * deriv_out);
        cuda_memcpy_to_device(select_d1, d_select_d1, select_d1_size);
        cuda_memcpy_to_device(select_d2, d_select_d2, select_d2_size);
        cuda_memcpy_to_device(select_col, d_select_col, select_rows * 16);

        record_cuda("add", bench_binary(cuda_add, d_a, d_b, d_out, size, repeats));
        record_cuda("sub", bench_binary(cuda_sub, d_a, d_b, d_out, size, repeats));
        record_cuda("mul", bench_binary(cuda_mult, d_a, d_b, d_out, size, repeats));
        record_cuda("mse", bench_binary(cuda_mse, d_a, d_b, d_out, size, repeats));
        record_cuda("mean", bench_unary(cuda_mean, d_a, d_out, size, repeats));
        record_cuda("square", bench_unary(cuda_square, d_a, d_out, size, repeats));
        record_cuda("tanh", bench_unary(cuda_tanh, d_a, d_out, size, repeats));
        record_cuda("scalar_mult", bench_scalar(cuda_scalar_mult, d_a, 2.0f, d_out, size, repeats));
        record_cuda("scalar_add", bench_scalar(cuda_scalar_add, d_a, 2.0f, d_out, size, repeats));
        record_cuda("identity", bench_unary(cuda_identity, d_a, d_out, size, repeats));
        record_cuda("matmult", bench_matmult(cuda_matmult, d_mat_a, d_mat_b, d_mat_out, rows, inner, cols, matrix_repeats));
        record_cuda("bias_add", bench_bias(cuda_bias_add, d_mat_a, d_bias, d_mat_out, rows, cols, repeats));
        record_cuda("scale_deriv", bench_scale(cuda_scale_deriv, d_scale_deriv, d_scale_factor, input_dim, d_scale_out, scale_size, repeats));
        record_cuda("chain_d2", bench_chain(cuda_chain_d2, d_chain_d1, d_chain_d2, d_chain_f, d_chain_fd, input_dim, chain_size, d_chain_out, repeats));
        record_cuda("deriv_matmult", bench_deriv(cuda_deriv_matmult, d_deriv1, d_W, batch, deriv_in, deriv_out, input_dim, d_deriv1_out, matrix_repeats));
        record_cuda("deriv2_matmult", bench_deriv(cuda_deriv2_matmult, d_deriv2, d_W, batch, deriv_in, deriv_out, input_dim, d_deriv2_out, matrix_repeats));
        record_cuda("select_d1", bench_select_d1(cuda_select_d1, d_select_d1, input_dim, 1, d_select_out, select_d1_size, repeats));
        record_cuda("select_d2", bench_select_d2(cuda_select_d2, d_select_d2, input_dim, 1, 1, d_select_out, select_d2_size, repeats));
        record_cuda("select_col", bench_select_col(cuda_select_col, d_select_col, 1, d_col_out, select_rows, 16, repeats));
        record_cuda("relu", bench_unary(cuda_relu, d_a, d_out, size, repeats));
        record_cuda("sigmoid", bench_unary(cuda_sigmoid, d_a, d_out, size, repeats));

        cuda_free(d_a); cuda_free(d_b); cuda_free(d_out);
        cuda_free(d_mat_a); cuda_free(d_mat_b); cuda_free(d_mat_out); cuda_free(d_bias);
        cuda_free(d_scale_deriv); cuda_free(d_scale_factor); cuda_free(d_scale_out);
        cuda_free(d_chain_d1); cuda_free(d_chain_d2); cuda_free(d_chain_f); cuda_free(d_chain_fd); cuda_free(d_chain_out);
        cuda_free(d_deriv1); cuda_free(d_deriv2); cuda_free(d_W); cuda_free(d_deriv1_out); cuda_free(d_deriv2_out);
        cuda_free(d_select_d1); cuda_free(d_select_d2); cuda_free(d_select_out); cuda_free(d_select_col); cuda_free(d_col_out);
    }
#endif

    print_results(has_cuda);

    free(a); free(b); free(out);
    free(mat_a); free(mat_b); free(mat_out); free(bias);
    free(scale_deriv); free(scale_factor); free(scale_out);
    free(chain_d1); free(chain_d2); free(chain_f); free(chain_fd); free(chain_out);
    free(deriv1); free(deriv2); free(W); free(deriv1_out); free(deriv2_out);
    free(select_d1); free(select_d2); free(select_out); free(select_col); free(col_out);
    return 0;
}
