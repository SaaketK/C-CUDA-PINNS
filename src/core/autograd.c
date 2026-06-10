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