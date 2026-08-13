/* Standalone serial C vs OpenMP vs OpenBLAS kernel benchmark. */
#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <omp.h>

static double now_ms(void){
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000.0 + time.tv_nsec / 1e6;
}

static void fill(float *data, int size, int offset){
    for(int i = 0; i < size; i++) data[i] = (float)((i + offset) % 37 - 18) / 19.0f;
}

static void tanh_c(const float *a, float *out, int size){ for(int i = 0; i < size; i++) out[i] = tanhf(a[i]); }
static void tanh_omp(const float *a, float *out, int size){
    #pragma omp parallel for
    for(int i = 0; i < size; i++) out[i] = tanhf(a[i]);
}

static void chain_c(const float *d1, const float *d2, const float *fp, const float *fpp, float *out, int batch, int dim){
    for(int i = 0; i < batch; i++) for(int p = 0; p < dim; p++) for(int q = 0; q < dim; q++){
        int h = i * dim * dim + p * dim + q;
        out[h] = fpp[i] * d1[i * dim + p] * d1[i * dim + q] + fp[i] * d2[h];
    }
}
static void chain_omp(const float *d1, const float *d2, const float *fp, const float *fpp, float *out, int batch, int dim){
    #pragma omp parallel for collapse(3)
    for(int i = 0; i < batch; i++) for(int p = 0; p < dim; p++) for(int q = 0; q < dim; q++){
        int h = i * dim * dim + p * dim + q;
        out[h] = fpp[i] * d1[i * dim + p] * d1[i * dim + q] + fp[i] * d2[h];
    }
}

static void deriv_c(const float *d, const float *w, float *out, int batch, int in, int out_features, int channels){
    for(int r = 0; r < batch; r++) for(int c = 0; c < out_features; c++) for(int ch = 0; ch < channels; ch++){
        float sum = 0.0f;
        for(int k = 0; k < in; k++) sum += d[(r * in + k) * channels + ch] * w[k * out_features + c];
        out[(r * out_features + c) * channels + ch] = sum;
    }
}
static void deriv_omp(const float *d, const float *w, float *out, int batch, int in, int out_features, int channels){
    #pragma omp parallel for collapse(3)
    for(int r = 0; r < batch; r++) for(int c = 0; c < out_features; c++) for(int ch = 0; ch < channels; ch++){
        float sum = 0.0f;
        for(int k = 0; k < in; k++) sum += d[(r * in + k) * channels + ch] * w[k * out_features + c];
        out[(r * out_features + c) * channels + ch] = sum;
    }
}

static void matmul_c(const float *a, const float *b, float *out, int rows, int inner, int cols){
    for(int r = 0; r < rows; r++) for(int c = 0; c < cols; c++){
        float sum = 0.0f;
        for(int k = 0; k < inner; k++) sum += a[r * inner + k] * b[k * cols + c];
        out[r * cols + c] = sum;
    }
}
static void matmul_omp(const float *a, const float *b, float *out, int rows, int inner, int cols){
    #pragma omp parallel for collapse(2)
    for(int r = 0; r < rows; r++) for(int c = 0; c < cols; c++){
        float sum = 0.0f;
        for(int k = 0; k < inner; k++) sum += a[r * inner + k] * b[k * cols + c];
        out[r * cols + c] = sum;
    }
}
static void matmul_blas(const float *a, const float *b, float *out, int rows, int inner, int cols){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, inner, 1.0f, a, inner, b, cols, 0.0f, out, cols);
}

typedef void (*Kernel)(void);
static double measure(Kernel kernel, int repeats){
    kernel();
    double start = now_ms();
    for(int i = 0; i < repeats; i++) kernel();
    return (now_ms() - start) / repeats;
}

static float *g_a, *g_b, *g_out, *g_d1, *g_d2, *g_fp, *g_fpp;
static int g_size, g_batch, g_dim, g_in, g_out_features, g_channels;
static void run_tanh_c(void){ tanh_c(g_a, g_out, g_size); }
static void run_tanh_omp(void){ tanh_omp(g_a, g_out, g_size); }
static void run_chain_c(void){ chain_c(g_d1, g_d2, g_fp, g_fpp, g_out, g_batch, g_dim); }
static void run_chain_omp(void){ chain_omp(g_d1, g_d2, g_fp, g_fpp, g_out, g_batch, g_dim); }
static void run_deriv_c(void){ deriv_c(g_a, g_b, g_out, g_batch, g_in, g_out_features, g_channels); }
static void run_deriv_omp(void){ deriv_omp(g_a, g_b, g_out, g_batch, g_in, g_out_features, g_channels); }
static void run_matmul_c(void){ matmul_c(g_a, g_b, g_out, g_batch, g_in, g_out_features); }
static void run_matmul_omp(void){ matmul_omp(g_a, g_b, g_out, g_batch, g_in, g_out_features); }
static void run_matmul_blas(void){ matmul_blas(g_a, g_b, g_out, g_batch, g_in, g_out_features); }

static void report(FILE *file, const char *name, int batch, int channels, Kernel serial, Kernel omp, Kernel blas, int repeats){
    double c = measure(serial, repeats), parallel = measure(omp, repeats), openblas = blas ? measure(blas, repeats) : 0.0;
    fprintf(file, "%-18s batch=%-5d channels=%-2d serial=%8.4f ms  openmp=%8.4f ms (%5.2fx)", name, batch, channels, c, parallel, c / parallel);
    if(blas) fprintf(file, "  openblas=%8.4f ms (%5.2fx)", openblas, c / openblas);
    fprintf(file, "\n");
}

int main(void){
    const int batches[] = {64, 128, 200, 512, 1024};
    FILE *file = fopen("src/tests/benchmark_openmp_vs_C_results.txt", "w");
    if(!file) return 1;
    fprintf(file, "OpenMP vs serial C vs OpenBLAS benchmark\nthreads=%d\n\n", omp_get_max_threads());
    for(int i = 0; i < 5; i++){
        g_batch = batches[i]; g_dim = 2; g_in = 64; g_out_features = 64;
        g_size = g_batch * g_in; g_a = malloc(g_batch * g_in * 4 * sizeof(float)); g_b = malloc(g_in * g_out_features * sizeof(float)); g_out = malloc(g_batch * g_out_features * 4 * sizeof(float));
        g_d1 = malloc(g_batch * g_dim * sizeof(float)); g_d2 = malloc(g_batch * g_dim * g_dim * sizeof(float)); g_fp = malloc(g_batch * sizeof(float)); g_fpp = malloc(g_batch * sizeof(float));
        fill(g_a, g_batch * g_in * 4, 1); fill(g_b, g_in * g_out_features, 2); fill(g_d1, g_batch * g_dim, 3); fill(g_d2, g_batch * g_dim * g_dim, 4); fill(g_fp, g_batch, 5); fill(g_fpp, g_batch, 6);
        fprintf(file, "batch=%d\n", g_batch);
        report(file, "tanh", g_size, 1, run_tanh_c, run_tanh_omp, NULL, 200);
        report(file, "chain_d2", g_batch, 4, run_chain_c, run_chain_omp, NULL, 200);
        g_channels = 2; report(file, "deriv_matmult", g_batch, 2, run_deriv_c, run_deriv_omp, NULL, 50);
        g_channels = 4; report(file, "deriv2_matmult", g_batch, 4, run_deriv_c, run_deriv_omp, NULL, 50);
        report(file, "matmult", g_batch, 1, run_matmul_c, run_matmul_omp, run_matmul_blas, 50);
        fprintf(file, "\n");
        free(g_a); free(g_b); free(g_out); free(g_d1); free(g_d2); free(g_fp); free(g_fpp);
    }
    fclose(file);
    return 0;
}
