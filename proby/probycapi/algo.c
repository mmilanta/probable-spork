#include <stdlib.h>
#include <stdio.h>
#include "algo.h"

unsigned int WIN = 0;
unsigned int LOSE = 1;

int Pow2(unsigned int x){
    int number = 1;
    for (unsigned int i = 0; i < x; ++i)
        number *= 2;
    return number;
}


unsigned int* nav(unsigned int *graph, unsigned int id) {
    // Navigate to the node with the given ID
    int k = graph[0];
    if (id >= graph[1]) {
        perror("Error: Node ID out of bounds");
    }
    return graph + 2 + ((id - 2) * Pow2(k));
}
int n_edges(unsigned int *graph) {
    // Number of outward edges, aways a power of 2
    return Pow2(graph[0]);
}
int n_nodes(unsigned int *graph) {
    // Number of nodes considering the two special nodes (WIN and LOSE)
    return graph[1];
}
typedef struct Monome {
    unsigned int id;
    double coeff;
    struct Monome* next;
} Monome;
Monome *sum_polynome(Monome *a, Monome *b, double b_coeff) {
    Monome *out = NULL;
    Monome *current = NULL;
    while (a != NULL || b != NULL) {
        Monome *new = (Monome *)malloc(sizeof(Monome));
        new->next = NULL;
        if (a == NULL) {
            new->id = b->id;
            new->coeff = b->coeff * b_coeff;
            b = b->next;
        } else if (b == NULL) {
            new->id = a->id;
            new->coeff = a->coeff;
            a = a->next;
        } else if (a->id == b->id) {
            new->id = a->id;
            new->coeff = a->coeff + (b->coeff * b_coeff);
            a = a->next;
            b = b->next;
        } else if (a->id < b->id) {
            new->id = a->id;
            new->coeff = a->coeff;
            a = a->next;
        } else {
            new->id = b->id;
            new->coeff = b->coeff * b_coeff;
            b = b->next;
        }
        if (out == NULL) {
            out = new;
            current = new;
        } else {
            current->next = new;
            current = new;
        }
    }
    return out;
}
void free_polynome(Monome *a) {
    Monome *next;
    while (a != NULL) {
        next = a->next;
        free(a);
        a = next;
    }
}
Monome * new_polynome(unsigned int id, double coeff) {
    Monome *out = (Monome *)malloc(sizeof(Monome));
    out->id = id;
    out->coeff = coeff;
    out->next = NULL;
    return out;
}
Monome *get_by_id(Monome *a, unsigned int id) {
    while (a != NULL) {
        if (a->id == id)
            return a;
        a = a->next;
    }
    return NULL;
}
int is_semplified(Monome *a) {
    while (a != NULL) {
        if (a->coeff != 0 && a->id != WIN)
            return 0;
        a = a->next;
    }
    return 1;
}


double* edge_probability(double * ps, unsigned int k){
    double *edge_probabilites = malloc(Pow2(k) * sizeof(double));
    for (int i = 0; i < Pow2(k); i++) {
        edge_probabilites[i] = 1;
        for (unsigned int j = 0; j < k; j++) {
            if (((i >> j) & 1) == 1)
                edge_probabilites[i] *= ps[k - j - 1];
            else
                edge_probabilites[i] *= 1 - ps[k - j - 1];
        }
    }
    return edge_probabilites;
}


void _prob(unsigned int graph[], char visiting[], Monome **prob_cache, double * edge_probabilites, unsigned int id) {
    // Recursively calculate the probability of winning from each node
    if (visiting[id] == 1){
        prob_cache[id] = new_polynome(id, 1);
        return;
    }
    if (id == WIN)
        return;
    if (id == LOSE)
        return;
    if (prob_cache[id] != NULL && is_semplified(prob_cache[id]))
        return;
    visiting[id] = 1;
    unsigned int *node = nav(graph, id);
    Monome *p = NULL;
    Monome *pn = NULL;
    for (int i = 0; i < n_edges(graph); i++) {
        _prob(graph, visiting, prob_cache, edge_probabilites, node[i]);
        pn = sum_polynome(p, prob_cache[node[i]], edge_probabilites[i]);
        free_polynome(p);
        p = pn;
    }
    Monome * p_id = get_by_id(p, id);
    Monome * p_next = p;
    if (p_id != NULL) {
        double p_id_coeff = get_by_id(p_next, id)->coeff;
        while (p_next != NULL) {
            if (p_next->id != id)
                p_next->coeff /= (1 - p_id_coeff);
            else
                p_next->coeff = 0;
            p_next = p_next->next;
        }
    }
    if (prob_cache[id] != NULL)
        free_polynome(prob_cache[id]);
    prob_cache[id] = p;
    visiting[id] = 0;
    return;
}


double prob(unsigned int *graph, double *ps, int index) {
    // Calculate the probability of winning.
    double *edge_probabilites = edge_probability(ps, graph[0]);
    Monome **prob_cache = malloc(n_nodes(graph)  * sizeof(Monome *));
    char *visiting = malloc(n_nodes(graph) * sizeof(char));
    prob_cache[WIN] = new_polynome(WIN, 1);
    prob_cache[LOSE] = new_polynome(LOSE, 0);
    for (int j = 0; j < n_nodes(graph); j++) {
        visiting[j] = 0;
    }

    for (int i = 2; i < n_nodes(graph); i++) {
        prob_cache[i] = NULL;
    }
    _prob(graph, visiting, prob_cache, edge_probabilites, index);
    double out = get_by_id(prob_cache[index], WIN)->coeff;
    for (int i = 0; i < n_nodes(graph); i++) {
        free_polynome(prob_cache[i]);
    }
    free(visiting);
    free(edge_probabilites);
    free(prob_cache);
    return out;
}

void _explen(unsigned int graph[], char visiting[], Monome **explen_cache, double * edge_probabilites, unsigned int id) {
    // Recursively calculate the probability of winning from each node
    if (visiting[id] == 1){
        explen_cache[id] = new_polynome(id, 1);
        return;
    }
    if (id == WIN)
        return;
    if (id == LOSE)
        return;
    if (explen_cache[id] != NULL && is_semplified(explen_cache[id]))
        return;
    visiting[id] = 1;
    unsigned int *node = nav(graph, id);
    Monome *p = new_polynome(WIN, 1);
    Monome *pn = NULL;
    for (int i = 0; i < n_edges(graph); i++) {
        _explen(graph, visiting, explen_cache, edge_probabilites, node[i]);
        pn = sum_polynome(p, explen_cache[node[i]], edge_probabilites[i]);
        free_polynome(p);
        p = pn;
    }
    Monome * p_id = get_by_id(p, id);
    Monome * p_next = p;
    if (p_id != NULL) {
        double p_id_coeff = get_by_id(p_next, id)->coeff;
        while (p_next != NULL) {
            if (p_next->id != id)
                p_next->coeff /= (1 - p_id_coeff);
            else
                p_next->coeff = 0;
            p_next = p_next->next;
        }
    }
    if (explen_cache[id] != NULL)
        free_polynome(explen_cache[id]);
    explen_cache[id] = p;
    visiting[id] = 0;
    return;
}


double explen(unsigned int *graph, double *ps, int index) {
    // Calculate the probability of winning.
    double *edge_probabilites = edge_probability(ps, graph[0]);
    Monome **explen_cache = malloc(n_nodes(graph)  * sizeof(Monome *));
    char *visiting = malloc(n_nodes(graph) * sizeof(char));
    explen_cache[WIN] = new_polynome(WIN, 0);
    explen_cache[LOSE] = new_polynome(LOSE, 0);
    for (int j = 0; j < n_nodes(graph); j++) {
        visiting[j] = 0;
    }

    for (int i = 2; i < n_nodes(graph); i++) {
        explen_cache[i] = NULL;
    }
    _explen(graph, visiting, explen_cache, edge_probabilites, index);
    double out = get_by_id(explen_cache[index], WIN)->coeff;
    for (int i = 0; i < n_nodes(graph); i++) {
        free_polynome(explen_cache[i]);
    }
    free(visiting);
    free(edge_probabilites);
    free(explen_cache);
    return out;
}
