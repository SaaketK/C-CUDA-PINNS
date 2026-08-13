/*
 * src/tests/sampler-test.c
 *
 * Smoke tests for domain and collocation-point sampling utilities.
 */

#include <stdio.h>
#include <stdlib.h>
#include "pinn/core/tensor.h"
#include "pinn/pinn/sampler.h"

static int test_uniform_box_shape_and_bounds(void){
    float lower[2] = {0.0f, -2.0f};
    float upper[2] = {1.0f, 3.0f};
    BoxDomain domain = {
        .dim = 2,
        .lower = lower,
        .upper = upper
    };
    int n_points = 100;
    Tensor *points = sample_uniform_box(&domain, n_points);
    int ok = 1;

    if(points->ndim != 2 || points->shape[0] != n_points || points->shape[1] != domain.dim){
        printf("uniform box shape FAILED: got shape=[%d, %d]\n", points->shape[0], points->shape[1]);
        ok = 0;
    }

    for(int i = 0; i < n_points; i++){
        for(int j = 0; j < domain.dim; j++){
            float value = points->data[i * domain.dim + j];
            if(value < lower[j] || value > upper[j]){
                printf("uniform box bounds FAILED at point=%d dim=%d value=%f lower=%f upper=%f\n",
                       i, j, value, lower[j], upper[j]);
                ok = 0;
            }
        }
    }

    if(ok){
        printf("uniform box sampler passed shape and bounds checks\n");
    }

    tensor_free(points);
    return ok;
}

static int test_fixed_dim_box_shape_and_bounds(void){
    float lower[2] = {0.0f, -2.0f};
    float upper[2] = {1.0f, 3.0f};
    BoxDomain domain = {
        .dim = 2,
        .lower = lower,
        .upper = upper
    };
    int n_points = 100;
    int fixed_dim = 1;
    float fixed_value = -2.0f;
    Tensor *points = sample_fixed_dim_box(&domain, n_points, fixed_dim, fixed_value);
    int ok = 1;

    if(points->ndim != 2 || points->shape[0] != n_points || points->shape[1] != domain.dim){
        printf("fixed-dim box shape FAILED: got shape=[%d, %d]\n", points->shape[0], points->shape[1]);
        ok = 0;
    }

    for(int i = 0; i < n_points; i++){
        for(int j = 0; j < domain.dim; j++){
            float value = points->data[i * domain.dim + j];
            if(j == fixed_dim){
                if(value != fixed_value){
                    printf("fixed-dim box fixed value FAILED at point=%d dim=%d value=%f expected=%f\n",
                           i, j, value, fixed_value);
                    ok = 0;
                }
            }
            else if(value < lower[j] || value > upper[j]){
                printf("fixed-dim box bounds FAILED at point=%d dim=%d value=%f lower=%f upper=%f\n",
                       i, j, value, lower[j], upper[j]);
                ok = 0;
            }
        }
    }

    if(ok){
        printf("fixed-dim box sampler passed shape, fixed-value, and bounds checks\n");
    }

    tensor_free(points);
    return ok;
}

static int test_lhs_box_shape_bounds_and_bins(void){
    float lower[2] = {-1.0f, 10.0f};
    float upper[2] = {1.0f, 20.0f};
    BoxDomain domain = {
        .dim = 2,
        .lower = lower,
        .upper = upper
    };
    int n_points = 25;
    Tensor *points = sample_LHS_box(&domain, n_points);
    int ok = 1;

    if(points->ndim != 2 || points->shape[0] != n_points || points->shape[1] != domain.dim){
        printf("LHS box shape FAILED: got shape=[%d, %d]\n", points->shape[0], points->shape[1]);
        ok = 0;
    }

    for(int d = 0; d < domain.dim; d++){
        int *counts = calloc(n_points, sizeof(int));
        if(!counts){
            printf("LHS test allocation FAILED\n");
            tensor_free(points);
            return 0;
        }

        float width = upper[d] - lower[d];
        for(int i = 0; i < n_points; i++){
            float value = points->data[i * domain.dim + d];
            if(value < lower[d] || value > upper[d]){
                printf("LHS bounds FAILED at point=%d dim=%d value=%f lower=%f upper=%f\n",
                       i, d, value, lower[d], upper[d]);
                ok = 0;
                continue;
            }

            int bin = (int)(((value - lower[d]) / width) * n_points);
            if(bin == n_points){
                bin = n_points - 1;
            }
            if(bin < 0 || bin >= n_points){
                printf("LHS bin index FAILED at point=%d dim=%d value=%f bin=%d\n",
                       i, d, value, bin);
                ok = 0;
            }
            else {
                counts[bin]++;
            }
        }

        for(int bin = 0; bin < n_points; bin++){
            if(counts[bin] != 1){
                printf("LHS bin count FAILED dim=%d bin=%d count=%d expected=1\n",
                       d, bin, counts[bin]);
                ok = 0;
            }
        }

        free(counts);
    }

    if(ok){
        printf("LHS box sampler passed shape, bounds, and one-sample-per-bin checks\n");
    }

    tensor_free(points);
    return ok;
}

int main(void){
    srand(0);
    if(!test_uniform_box_shape_and_bounds()){
        printf("sampler tests failed\n");
        return 1;
    }
    if(!test_fixed_dim_box_shape_and_bounds()){
        printf("sampler tests failed\n");
        return 1;
    }
    if(!test_lhs_box_shape_bounds_and_bins()){
        printf("sampler tests failed\n");
        return 1;
    }

    printf("all sampler tests passed\n");
    return 0;
}
