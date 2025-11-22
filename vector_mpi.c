#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    const long N = 200 000000;  // Общее количество элементов
    int rank, size;
    double local_sum = 0.0, global_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long chunk = N / size;
    
    double *A = malloc(chunk * sizeof(double));
    double *B = malloc(chunk * sizeof(double));

    double *A_full = NULL;
    double *B_full = NULL;
    if (rank == 0) {
        A_full = malloc(N * sizeof(double));
        B_full = malloc(N * sizeof(double));

        for (long i = 0; i < N; i++) {
            A_full[i] = (double)i;  //от 0 до N-1
            B_full[i] = (double)(N - i - 1);  //  от N-1 до 0
        }
    }

    // Используем MPI_Scatter для распределения данных
    MPI_Scatter(A_full, chunk, MPI_DOUBLE, A, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_full, chunk, MPI_DOUBLE, B, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    // Вычисление локальной суммы скалярного произведения
    
    for (long i = 0; i < chunk; i++) {
        local_sum += A[i] * B[i];
    }

    // Сбор всех локальных сумм в процессе 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    // Только процесс 0 выводит результат
    if (rank == 0) {
        printf("MPI: result = %.2f, time = %f sec, processes = %d\n", global_sum, t2 - t1, size);
    }

    // Освобождаем память
    if (rank == 0) {
        free(A_full);
        free(B_full);
    }
    free(A);
    free(B);

    MPI_Finalize();
    return 0;
}
