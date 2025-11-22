#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>  

#define N 10000000

int main() {
    int *A = (int *)malloc(N * sizeof(int));
    int *B = (int *)malloc(N * sizeof(int));
    int *C = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    clock_t start = clock();  
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        C[i] = (A[i] + B[i]) * (A[i] - B[i]);
    }
    clock_t end = clock();  

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Parallel time: %f seconds\n", time_taken);

    free(A);
    free(B);
    free(C);

    return 0;
}
