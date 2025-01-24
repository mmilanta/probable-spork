#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>


unsigned int* fetch_data(){
    const char *path = "serialized_graph.bin";
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening file");
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    printf("File size: %ld\n", file_size);
    // Ensure file size is a multiple of 4
    if (file_size % 4 != 0) {
        fprintf(stderr, "Error: File size is not a multiple of 4\n");
        fclose(file);
    }

    // Allocate buffer for file data
    uint8_t *buf = malloc(file_size);
    if (buf == NULL) {
        perror("Error allocating memory");
        fclose(file);
    }

    // Read file data into buffer
    size_t read_size = fread(buf, 1, file_size, file);
    if (read_size != file_size) {
        perror("Error reading file");
        free(buf);
        fclose(file);
    }
    fclose(file);
    // Cast buffer to u32 array
    size_t num_u32 = file_size / 4;
    unsigned int *u32_array = (unsigned int *)buf;

    // Print the u32 values
    for (unsigned int i = 0; i < num_u32; i++) {
        //printf("%u: %u\n", i, u32_array[i]);
    }
    return u32_array;
}


unsigned int WIN = 0;
unsigned int LOSE = 1;

int Pow2(unsigned int x){
    int number = 1;
    for (int i = 0; i < x; ++i)
        number *= 2;
    return number;
}


unsigned int* nav(unsigned int *graph, int id) {
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
void _prob(unsigned int graph[], char visiting[], double **prob_cache, double * edge_probabilites, unsigned int id) {
    if (visiting[id] == 1){
        prob_cache[id][0] = 0;
        prob_cache[id][id] = 1;
    }
    if (id == WIN)
        return;
    if (id == LOSE)
        return;
    if (prob_cache[id][0] >= -.5)
        return;

    visiting[id] = 1;
    unsigned int *node = nav(graph, id);
    double *p = malloc(n_nodes(graph) * sizeof(double));
    for (int j = 0; j < n_nodes(graph); j++) {
        p[j] = 0;
    }
    for (int i = 0; i < n_edges(graph); i++) {
        _prob(graph, visiting, prob_cache, edge_probabilites, node[i]);
        for (int j = 0; j < n_nodes(graph); j++) {
            p[j] += edge_probabilites[i] * prob_cache[node[i]][j];
        }
    }
    for (int j = 0; j < n_nodes(graph); j++) {
        prob_cache[id][j] = p[j] / (1 - p[id]);
    }

    free(p);
    visiting[id] = 0;
    return;
}


double prob(unsigned int *graph, double *ps) {
    double *edge_probabilites = malloc(n_edges(graph) * sizeof(double));
    for (int i = 0; i < n_edges(graph); i++) {
        edge_probabilites[i] = 1;
        for (int j = 0; j < graph[0]; j++) {
            if (((i >> j) & 1) == 1)
                edge_probabilites[i] *= ps[graph[0] - j - 1];
            else
                edge_probabilites[i] *= 1 - ps[graph[0] - j - 1];
        }
    }

    double **prob_cache = malloc(n_nodes(graph)  * sizeof(double *));
    char *visiting = malloc(n_nodes(graph) * sizeof(char));
    prob_cache[WIN] = malloc(n_nodes(graph)  * sizeof(double));
    prob_cache[LOSE] = malloc(n_nodes(graph)  * sizeof(double));
    for (int j = 0; j < n_nodes(graph); j++) {
        visiting[j] = 0;
        prob_cache[WIN][j] = 0;
        prob_cache[LOSE][j] = 0;
    }
    prob_cache[WIN][0] = 1;

    for (int i = 2; i < n_nodes(graph); i++) {
        prob_cache[i] = malloc(n_nodes(graph)  * sizeof(double));
        prob_cache[i][0] = -1; // Cache is empty
        for (int j = 1; j < n_nodes(graph); j++) {
            prob_cache[i][j] = 0;
        }
    }
    _prob(graph, visiting, prob_cache, edge_probabilites, 2);
    double out = prob_cache[2][0];
    free(edge_probabilites);
    free(prob_cache);
    return out;
}

int main() {

    unsigned int *graph = fetch_data();
    double p;
    double x;

    for (double i = 5; i < 10; i++) {
        p = 0.5 + i/1000;
        x = prob(graph, (double[]){p});
        printf("%f -> %f\n", p, x);
    }
}