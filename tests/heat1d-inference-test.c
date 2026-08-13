#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pinn/pinn/residual.h"
#include "pinn/surrogate/heat1d_inference.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef PINN_TEST_MODEL_PATH
#define PINN_TEST_MODEL_PATH "models/heat1d/v1/model.pinn"
#endif

typedef struct {
    float alpha;
    float a1;
    float a2;
} ValidationCase;

static float exact(float t, float x, float alpha, float a1, float a2){
    float pi2 = (float)M_PI * (float)M_PI;
    return a1 * sinf((float)M_PI * x) * expf(-alpha * pi2 * t)
        + a2 * sinf(2.0f * (float)M_PI * x)
            * expf(-4.0f * alpha * pi2 * t);
}

static float evaluate_case(
    Heat1DInference *inference,
    ValidationCase validation_case
){
    const int nt = 101;
    const int nx = 101;
    const int count = nt * nx;
    float *points = malloc((size_t)count * HEAT1D_INPUT_DIM * sizeof(float));
    float *prediction = malloc((size_t)count * sizeof(float));
    if(!points || !prediction){
        free(points);
        free(prediction);
        return INFINITY;
    }

    for(int it = 0; it < nt; it++){
        float t = (float)it / (float)(nt - 1);
        for(int ix = 0; ix < nx; ix++){
            float x = (float)ix / (float)(nx - 1);
            int row = it * nx + ix;
            points[row * HEAT1D_INPUT_DIM + HEAT1D_T] = t;
            points[row * HEAT1D_INPUT_DIM + HEAT1D_X] = x;
            points[row * HEAT1D_INPUT_DIM + HEAT1D_ALPHA] = validation_case.alpha;
            points[row * HEAT1D_INPUT_DIM + HEAT1D_A1] = validation_case.a1;
            points[row * HEAT1D_INPUT_DIM + HEAT1D_A2] = validation_case.a2;
        }
    }

    if(!heat1d_inference_predict(inference, points, count, prediction)){
        free(points);
        free(prediction);
        return INFINITY;
    }

    double squared_error = 0.0;
    double squared_reference = 0.0;
    for(int row = 0; row < count; row++){
        float t = points[row * HEAT1D_INPUT_DIM + HEAT1D_T];
        float x = points[row * HEAT1D_INPUT_DIM + HEAT1D_X];
        float reference = exact(
            t, x, validation_case.alpha, validation_case.a1, validation_case.a2
        );
        double error = (double)prediction[row] - reference;
        squared_error += error * error;
        squared_reference += (double)reference * reference;
    }
    free(points);
    free(prediction);
    return (float)sqrt(squared_error / squared_reference);
}

int main(void){
    if(heat1d_inference_input_dim() != HEAT1D_INPUT_DIM){
        fprintf(stderr, "unexpected Heat 1D inference input dimension\n");
        return 1;
    }
    Heat1DInference *missing = heat1d_inference_load(
        PINN_TEST_MODEL_PATH ".missing"
    );
    if(missing){
        fprintf(stderr, "wrapper unexpectedly loaded a missing model\n");
        heat1d_inference_free(missing);
        return 1;
    }
    Heat1DInference *inference = heat1d_inference_load(PINN_TEST_MODEL_PATH);
    if(!inference){
        fprintf(stderr, "failed to load %s\n", PINN_TEST_MODEL_PATH);
        return 1;
    }

    const ValidationCase cases[] = {
        {0.10f,  1.00f,  0.00f},
        {0.50f,  0.00f,  1.00f},
        {0.05f,  0.70f, -0.40f},
        {0.20f, -0.60f,  0.25f},
        {0.50f, -1.00f,  1.00f},
    };
    int passed = 1;
    for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++){
        float relative_l2 = evaluate_case(inference, cases[i]);
        printf(
            "wrapper alpha=%g a1=%g a2=%g rel_l2=%e\n",
            cases[i].alpha, cases[i].a1, cases[i].a2, relative_l2
        );
        if(!isfinite(relative_l2) || relative_l2 > 1e-2f) passed = 0;
    }

    float invalid_point[HEAT1D_INPUT_DIM] = {0.5f, 0.5f, 0.75f, 1.0f, 0.0f};
    float invalid_output = 0.0f;
    if(heat1d_inference_predict(inference, invalid_point, 1, &invalid_output)){
        fprintf(stderr, "wrapper accepted an out-of-domain diffusivity\n");
        passed = 0;
    }

    heat1d_inference_free(inference);
    if(!passed) return 1;
    printf("Heat 1D deployment wrapper reconstruction passed\n");
    return 0;
}
