/* Stable C inference ABI for the deployed Heat 1D modal surrogate. */
#ifndef PINN_SURROGATE_HEAT1D_INFERENCE_H
#define PINN_SURROGATE_HEAT1D_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Heat1DInference Heat1DInference;

/* Load a serialized modal model. Returns NULL if the artifact is incompatible. */
Heat1DInference *heat1d_inference_load(const char *model_path);

/* Release a model returned by heat1d_inference_load. */
void heat1d_inference_free(Heat1DInference *inference);

/*
 * Predict a batch of physical points stored row-major as
 * [t, x, alpha, a1, a2]. The output contains one reconstructed u value per
 * row. Internally the shared q network is evaluated at both
 * alpha*pi^2*t and 4*alpha*pi^2*t.
 *
 * Returns 1 on success and 0 on invalid input or inference failure.
 */
int heat1d_inference_predict(
    Heat1DInference *inference,
    const float *physical_points,
    int point_count,
    float *output
);

/* Return the model's physical input dimension (currently five). */
int heat1d_inference_input_dim(void);

#ifdef __cplusplus
}
#endif

#endif
