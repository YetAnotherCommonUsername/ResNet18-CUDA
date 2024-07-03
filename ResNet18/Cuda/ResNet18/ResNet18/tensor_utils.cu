#include "tensor_utils.cuh"

// Function to free the memory allocated for the tensor
void free_tensor(struct tensor* t) {
    if (t->data) {
        free(t->data);
        t->data = NULL;
    }
}

// Function to initializate the tensor
void init_tensor(struct tensor* t) {
    int num_elements = t->row * t->col;

    for (int i = 0; i < t->depth; i++) {
        for (int j = 0; j < num_elements; j++) {
            t->data[i*num_elements + j] = (float)(i+1);
        }
    }
}

// Function to randomly initializate the tensor
void init_random_tensor(struct tensor* t) {
    int num_elements = t->row * t->col;

    const int seed = 111222333;
    srand(seed);

    for (int i = 0; i < t->depth; i++) {
        for (int j = 0; j < num_elements; j++) {
            t->data[i * num_elements + j] = static_cast<float>(rand()-rand()) / RAND_MAX;
        }
    }
}

// Function to print the tensor
void print_tensor(struct tensor* t) {
    int num_elements = t->row * t->col;

    printf("Shape of the tensor: (%d, %d, %d)\n", t->depth, t->row, t->col);
    printf("\n");

    for (int i = 0; i < t->depth; i++) {
        printf("[");
        for (int j = 0; j < t->row; j++) {
            printf("[");
            for (int k = 0; k < t->col; k++) {
                printf("%f\t", t->data[i * num_elements + j * t->col + k]);
            }
            printf("]\n");
        }
        printf("]\n");
    }
}