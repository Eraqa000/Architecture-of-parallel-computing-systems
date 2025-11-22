#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    const long N = 200000000; // 200 млн элементов
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double result = 0.0;

    // Заполнение массивов A и B числами от 0 до N-1
    for (long i = 0; i < N; i++) {
        A[i] = (double)i;       // Массив A - числа от 0 до N-1
        B[i] = (double)(N - i - 1); // Массив B - числа от N-1 до 0
    }

    double start = omp_get_wtime();

    // Параллельный цикл с OpenMP для вычисления скалярного произведения
    #pragma omp parallel for schedule(static, 1)
    for (long i = 0; i < N; i++) {
        result += A[i] * B[i];
    }

    double end = omp_get_wtime();

    printf("OpenMP: result = %.2f, time = %f sec\n", result, end - start);

    free(A);
    free(B);

    return 0;
}
