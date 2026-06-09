/*
 * core/autograd.c
 *
 * Implements reverse-mode autograd graph traversal: build topological order from
 * a scalar loss, seed loss gradients, call backward functions in reverse order,
 * and later manage dynamic topo storage and graph cleanup.
 */

#include <stdio.h>
#include <stdlib.h>
#include "pinn/core/tensor.h"

Node* topo_order[100];
int topo_size = 0;

int contains_node(Node* node){
    for(int i = 0; i < topo_size; i++){
        if(topo_order[i] == node){
            return 1;
        }
    }
    return 0;
}

void build_topo(Tensor* t){
    if(t == NULL || t->grad_fn == NULL){
        return;
    }
    Node* node = t->grad_fn;
    if(contains_node(node)) return;

    for(int i = 0; i < node->n_inputs; i++){
        build_topo(node->inputs[i]);
    }
    topo_order[topo_size++] = node;
}

void backward(Tensor *loss){
    topo_size = 0;
    build_topo(loss);
    loss->grad = 1.0f;
    for(int i = topo_size - 1; i >= 0; i--){
        topo_order[i]->backward(topo_order[i]);
    }
}
