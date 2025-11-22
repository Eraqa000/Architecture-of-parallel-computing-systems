#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000 // Размер векторов

int main() {
    int *A = (int *)malloc(N * sizeof(int));
    int *B = (int *)malloc(N * sizeof(int));
    int *C = (int *)malloc(N * sizeof(int));

    // Инициализация массивов случайными значениями
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    clock_t start = clock();  // Засекаем время начала
    // Последовательное сложение векторов
    for (int i = 0; i < N; i++) {
        C[i] = (A[i] + B[i]) * (A[i] - B[i]);
    }
    clock_t end = clock();  // Засекаем время конца

    // Выводим время выполнения
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequential time: %f seconds\n", time_taken);

    // Освобождение памяти
    free(A);
    free(B);
    free(C);

    return 0;
}
