#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int N;
    double *A = NULL, *B = NULL, *C = NULL;

    printf("Enter matrix size N: ");
    scanf("%d", &N);

    // Матрицаларды динамикалық бөлу
    A = malloc(N * N * sizeof(double));
    B = malloc(N * N * sizeof(double));
    C = malloc(N * N * sizeof(double));

    srand((unsigned)time(NULL));

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 20; 
        B[i] = rand() % 20; 
    }

    printf("\nMatrix A:\n");
    
    printf("\nMatrix B:\n");
    

    clock_t start_time = clock(); 

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    clock_t end_time = clock(); // уақытты аяқтау

    // Қорытынды нәтиже матрицасы C
    printf("\nResult matrix C = A * B:\n");
    
    // Орындау уақытын есептеу
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nComputation time: %f seconds\n", time_taken);


    return 0;
}
