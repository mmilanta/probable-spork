#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>


uint32_t* fetch_data(){
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
    uint32_t *u32_array = (uint32_t *)buf;

    // Print the u32 values
    for (uint32_t i = 0; i < num_u32; i++) {
        //printf("%u: %u\n", i, u32_array[i]);
    }
    return u32_array;
}


uint32_t WIN = 0;
uint32_t LOSE = 1;

int Pow2(uint32_t x){
    int number = 1;
    for (int i = 0; i < x; ++i)
        number *= 2;
    return number;
}


uint32_t* nav(uint32_t *graph, int id) {
    int k = graph[0];
    if (id >= graph[1]) {
        perror("Error: Node ID out of bounds");
    }
    return graph + 2 + ((id - 2) * Pow2(k));
}
int n_edges(uint32_t *graph) {
    return Pow2(graph[0]);
}
int n_nodes(uint32_t *graph) {
    return graph[1];
}
double _prob(uint32_t *graph, double* prob_cache, double * edge_probabilites, uint32_t id) {
    if (prob_cache[id] >= 0) {
        return prob_cache[id];
    }
    if (id == WIN) {
        return 1;
    } else if (id == LOSE) {
        return 0;
    } else {
        uint32_t *node = nav(graph, id);
        double prob = 0;
        for (int i = 0; i < n_edges(graph); i++) {
            prob += edge_probabilites[i] * _prob(graph, prob_cache, edge_probabilites, node[i]);
        }
        prob_cache[id] = prob;
        return prob;
    }
}


double prob(uint32_t *graph, double * ps) {
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

    double *prob_cache = malloc(n_nodes(graph) * sizeof(double));
    for (int i = 0; i < n_nodes(graph); i++) {
        prob_cache[i] = -1.0;
    }
    double out = _prob(graph, prob_cache, edge_probabilites, 2);
    free(edge_probabilites);
    free(prob_cache);
    return out;
}

int main() {

    uint32_t *graph = fetch_data();
    double x;
    for (double i = 0; i < 10; i++) {
        x = prob(graph, (double[]){0.5, 0.5, i/10});
        printf("%f -> %f\n", i/100, x);
    }
    printf("Probability: %f\n", x);
}