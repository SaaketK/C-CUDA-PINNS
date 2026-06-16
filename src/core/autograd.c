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
#include "pinn/core/autograd.h"

void nodelist_init(NodeList *list){
    list->items = NULL;
    list->size = 0;
    list->capacity = 0;
}

int nodelist_push(NodeList *list, Node *node){
    if(list->size == list->capacity){
        int new_capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        Node **new_items = realloc(list->items, new_capacity * sizeof(Node*));
        if(!new_items) return 0;

        list->items = new_items;
        list->capacity = new_capacity;
    }
    list->items[list->size] = node;
    list->size++;
    return 1;
}
int nodelist_contains(NodeList *list, Node *node){
    for(int i = 0; i < list->size; i++){
        if(list->items[i] == node){
            return 1;
        }
    }
    return 0;
}
void nodelist_free(NodeList *list){
    free(list->items);
    list->items = NULL;
    list->size = 0;
    list->capacity = 0;
}

static void build_topo(Tensor *t, NodeList *topo){
    if(t == NULL || t->grad_fn == NULL) return;

    Node *node = t->grad_fn;
    if(nodelist_contains(topo, node)){
        return;
    }
    for(int i = 0; i < node->n_inputs; i++){
        build_topo(node->inputs[i], topo);
    }
    nodelist_push(topo, node);
}

void backward(Tensor *loss){
    if(!loss) return;
    if(loss->size != 1) return;

    NodeList topo;
    nodelist_init(&topo);
    build_topo(loss, &topo);    

    if(loss->grad == NULL){
        loss->grad = calloc(loss->size, sizeof(float));
        if(loss->grad == NULL){
            nodelist_free(&topo);
            return;
        }
    }
    loss->grad[0] = 1.0f;
    for(int i = topo.size - 1; i >= 0; i--){
        topo.items[i]->backward(topo.items[i]);
    }
    nodelist_free(&topo);
}

// Node free using Tape

static Tape *current_tape = NULL;

static void node_free(Node *node){
    if(!node) return;
    if(node->free_ctx) node->free_ctx(node->ctx);
    free(node->inputs);
    free(node);
}

Tape* tape_create(void){
    Tape *tape = malloc(sizeof(Tape));
    if(!tape) return NULL;
    tape->nodes = NULL;
    tape->tensors = NULL;
    tape->n_nodes = 0;
    tape->node_capacity = 0;
    tape->n_tensors = 0;
    tape->tensor_capacity = 0;
    return tape;
}

int tape_add_node(Tape *tape, Node *node){
    if(!tape || !node) return 0;

    if(tape->n_nodes == tape->node_capacity){
        int new_capacity = tape->node_capacity == 0 ? 16 : tape->node_capacity * 2;
        Node **new_nodes = realloc(tape->nodes, new_capacity * sizeof(Node*));
        if(!new_nodes) return 0;
        tape->nodes = new_nodes;
        tape->node_capacity = new_capacity;
    }
    tape->nodes[tape->n_nodes] = node;
    tape->n_nodes++;
    return 1;
}

int tape_add_tensor(Tape *tape, Tensor *tensor){
    if(!tape || !tensor) return 0;

    if(tape->n_tensors == tape->tensor_capacity){
        int new_capacity = tape->tensor_capacity == 0 ? 16 : tape->tensor_capacity * 2;
        Tensor **new_tensors = realloc(tape->tensors, new_capacity * sizeof(Tensor*));
        if(!new_tensors) return 0;

        tape->tensors = new_tensors;
        tape->tensor_capacity = new_capacity;
    }

    tape->tensors[tape->n_tensors] = tensor;
    tape->n_tensors++;
    return 1;
}
Tape* get_curr_tape(){
    return current_tape;
}
void set_curr_tape(Tape *tape){
    current_tape = tape;
}

void tape_free(Tape *tape){
    if(!tape) return;
    if(current_tape == tape){
        current_tape = NULL;
    }
    for(int i = 0; i < tape->n_nodes; i++){
        node_free(tape->nodes[i]);
    }
    free(tape->nodes);
    for(int i = 0; i < tape->n_tensors; i++){
        tensor_free(tape->tensors[i]);
    }
    free(tape->tensors);
    free(tape);
}
