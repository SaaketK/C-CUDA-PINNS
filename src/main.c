/*
 * src/main.c
 *
 * Placeholder executable for the framework. 
 * Smoke tests live under src/tests.
*/

#include <stdio.h>
#include <stdlib.h>
#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"

int main(void) {
    Tape *tape = tape_create();
    set_curr_tape(tape);

    if(get_curr_tape() != tape){
        printf("tape current pointer test failed\n");
        return 1;
    }

    for(int i = 0; i < 20; i++){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->n_inputs = 2;
        node->inputs[0] = NULL;
        node->inputs[1] = NULL;
        node->output = NULL;
        node->backward = NULL;
        tape_add_node(tape, node);

        int shape[1] = {1};
        Tensor *tensor = tensor_create(shape, 1, 1);
        tensor->data[0] = (float)i;
        tape_add_tensor(tape, tensor);
    }

    printf("tape before free: nodes=%d tensors=%d\n", tape->n_nodes, tape->n_tensors);
    tape_free(tape);
    printf("tape free smoke test passed\n");
    return 0;
}
