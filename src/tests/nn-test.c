/*
 * src/tests/nn-test.c
 *
 * Smoke tests for Linear, MLP, parameter collection, and SGD.
*/

#include <stdio.h>
#include <stdlib.h>
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/core/autograd.h"
#include "pinn/nn/mlp.h"
#include "pinn/nn/optimizer.h"

int main(void) {
    int mat_a_shape[2] = {2, 3};
    float mat_a_values[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Tensor *lin_x = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    Linear *linear = linear_create(3, 2);
    Tensor *lin_y = linear_forward(linear, lin_x);
    Tensor *lin_loss = tensor_mean(lin_y);
    backward(lin_loss);
    tensor_print(lin_y, "linear(lin_x)");
    tensor_print(lin_loss, "mean(linear(lin_x))");
    printf("linear W grad=[%f, %f, %f, %f, %f, %f]\n",
           linear->W->grad[0], linear->W->grad[1], linear->W->grad[2],
           linear->W->grad[3], linear->W->grad[4], linear->W->grad[5]);
    printf("linear b grad=[%f, %f]\n", linear->b->grad[0], linear->b->grad[1]);

    Tensor *mlp_x = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    int mlp_sizes[3] = {3, 4, 2};
    MLP *mlp = mlp_create(mlp_sizes, 3);
    Tensor *mlp_y = mlp_forward(mlp, mlp_x);
    Tensor *mlp_loss = tensor_mean(mlp_y);
    backward(mlp_loss);
    tensor_print(mlp_y, "mlp(mlp_x)");
    tensor_print(mlp_loss, "mean(mlp(mlp_x))");
    printf("mlp layer 0 W grad first row=[%f, %f, %f, %f]\n",
           mlp->layers[0]->W->grad[0], mlp->layers[0]->W->grad[1],
           mlp->layers[0]->W->grad[2], mlp->layers[0]->W->grad[3]);
    printf("mlp layer 0 b grad=[%f, %f, %f, %f]\n",
           mlp->layers[0]->b->grad[0], mlp->layers[0]->b->grad[1],
           mlp->layers[0]->b->grad[2], mlp->layers[0]->b->grad[3]);
    printf("mlp layer 1 W grad first row=[%f, %f]\n",
           mlp->layers[1]->W->grad[0], mlp->layers[1]->W->grad[1]);
    printf("mlp layer 1 b grad=[%f, %f]\n",
           mlp->layers[1]->b->grad[0], mlp->layers[1]->b->grad[1]);

    float train_target_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int train_target_shape[2] = {2, 2};
    Tensor *train_x = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    Tensor *train_y = tensor_from_data(train_target_values, train_target_shape, 2, 0);
    MLP *train_mlp = mlp_create(mlp_sizes, 3);
    int n_params = 0;
    Tensor **params = mlp_parameters(train_mlp, &n_params);

    Tensor *train_pred = mlp_forward(train_mlp, train_x);
    Tensor *train_loss = tensor_mse(train_pred, train_y);
    float before_step = train_loss->data[0];

    sgd_zero_grad(params, n_params);
    backward(train_loss);
    sgd_step(params, n_params, 0.01f);

    Tensor *after_pred = mlp_forward(train_mlp, train_x);
    Tensor *after_loss = tensor_mse(after_pred, train_y);
    float after_step = after_loss->data[0];

    printf("sgd smoke loss before=%f after=%f\n", before_step, after_step);

    int line_shape[2] = {5, 1};
    float line_x_values[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    float line_y_values[5] = {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    int line_sizes[3] = {1, 8, 1};

    Tensor *line_x = tensor_from_data(line_x_values, line_shape, 2, 1);
    Tensor *line_y = tensor_from_data(line_y_values, line_shape, 2, 0);
    MLP *line_mlp = mlp_create(line_sizes, 3);
    int line_n_params = 0;
    Tensor **line_params = mlp_parameters(line_mlp, &line_n_params);
    float first_line_loss = 0.0f;

    for(int step = 0; step < 500; step++){
        Tape *step_tape = tape_create();
        set_curr_tape(step_tape);
        sgd_zero_grad(line_params, line_n_params);

        Tensor *line_pred = mlp_forward(line_mlp, line_x);
        Tensor *line_loss = tensor_mse(line_pred, line_y);

        if(step == 0){
            first_line_loss = line_loss->data[0];
        }
        if(step % 100 == 0){
            printf("line train step=%d loss=%f\n", step, line_loss->data[0]);
        }

        backward(line_loss);
        sgd_step(line_params, line_n_params, 0.05f);

        tape_free(step_tape);
    }

    Tape *final_tape = tape_create();
    set_curr_tape(final_tape);
    Tensor *line_final_pred = mlp_forward(line_mlp, line_x);
    Tensor *line_final_loss = tensor_mse(line_final_pred, line_y);
    printf("line train loss first=%f final=%f\n", first_line_loss, line_final_loss->data[0]);
    tensor_print(line_final_pred, "line final predictions");
    tape_free(final_tape);
    tensor_free(line_y);
    tensor_free(line_x);
    free(line_params);
    mlp_free(line_mlp);

    Tensor *adam_x = tensor_from_data(line_x_values, line_shape, 2, 1);
    Tensor *adam_y = tensor_from_data(line_y_values, line_shape, 2, 0);
    MLP *adam_mlp = mlp_create(line_sizes, 3);
    int adam_n_params = 0;
    Tensor **adam_params = mlp_parameters(adam_mlp, &adam_n_params);
    Adam *adam = adam_create(adam_params, adam_n_params, 0.01f);
    float first_adam_loss = 0.0f;

    for(int step = 0; step < 500; step++){
        Tape *adam_tape = tape_create();
        set_curr_tape(adam_tape);
        adam_zero_grad(adam);

        Tensor *adam_pred = mlp_forward(adam_mlp, adam_x);
        Tensor *adam_loss = tensor_mse(adam_pred, adam_y);

        if(step == 0){
            first_adam_loss = adam_loss->data[0];
        }
        if(step % 100 == 0){
            printf("adam line train step=%d loss=%f\n", step, adam_loss->data[0]);
        }

        backward(adam_loss);
        adam_step(adam);

        tape_free(adam_tape);
    }

    Tape *adam_final_tape = tape_create();
    set_curr_tape(adam_final_tape);
    Tensor *adam_final_pred = mlp_forward(adam_mlp, adam_x);
    Tensor *adam_final_loss = tensor_mse(adam_final_pred, adam_y);
    printf("adam line train loss first=%f final=%f\n", first_adam_loss, adam_final_loss->data[0]);
    tensor_print(adam_final_pred, "adam line final predictions");
    tape_free(adam_final_tape);
    adam_free(adam);
    free(adam_params);
    tensor_free(adam_y);
    tensor_free(adam_x);
    mlp_free(adam_mlp);
    tensor_free(after_loss);
    tensor_free(after_pred);
    tensor_free(train_loss);
    tensor_free(train_pred);
    tensor_free(train_y);
    tensor_free(train_x);
    free(params);
    mlp_free(train_mlp);
    tensor_free(mlp_loss);
    tensor_free(mlp_y);
    tensor_free(mlp_x);
    mlp_free(mlp);
    tensor_free(lin_loss);
    tensor_free(lin_y);
    tensor_free(lin_x);
    linear_free(linear);

    return 0;
}
