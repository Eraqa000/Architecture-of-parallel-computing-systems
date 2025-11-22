/*
  vector_add_mpi_shrink_end.c

  Пример MPI-программы, которая на каждом шаге вычисляет суммы соседних пар
  и затем уменьшается (shrinks) по длине массива на root (rank 0) в конце итерации.

  Компиляция (MS-MPI / Windows, gcc/MSYS):
    gcc vector_add_mpi_shrink_end.c -I"C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Include" \
      -L"C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64" -lmsmpi -o vector_add_mpi_shrink_end.exe

  Запуск (пример, 4 процесса):
    mpiexec -n 4 vector_add_mpi_shrink_end.exe

  Файл выводит распределение задач на каждой итерации, локальные полученные элементы
  в каждом процессе и то, как root уменьшаеt (replaces) массив в конце итерации.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Демонстрационное значение. Можно изменить на входной параметр по желанию. */
    int N = 10;

    double *a = NULL;
    if (rank == 0) {
        a = (double*)malloc((size_t)N * sizeof(double));
        if (!a) { fprintf(stderr, "Allocation failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < N; ++i) a[i] = (double)(i + 1);

        printf("Initial array (n=%d):\n", N);
        for (int i = 0; i < N; ++i) printf("%.0f ", a[i]);
        printf("\n\n");
    }

    int *sendcounts = (int*)calloc(size, sizeof(int));
    int *displs = (int*)calloc(size, sizeof(int));
    int *recvcounts = (int*)calloc(size, sizeof(int));
    int *recvdispls = (int*)calloc(size, sizeof(int));

    double *local_a = NULL;
    double *local_c = NULL;

    int curr_n = N;
    int iteration = 0;
    double t0 = MPI_Wtime();

    while (curr_n > 1) {
        iteration++;
        int M = curr_n - 1; /* число сумм, которые нужно получить */

        /* Распределяем M сумм между процессами: recvcounts[] = сколько сумм вернуть */
        int base = M / size;
        int rem = M % size;
        int cum = 0;
        for (int i = 0; i < size; ++i) {
            int sums_count = base + (i < rem ? 1 : 0);
            recvcounts[i] = sums_count;           /* сколько сумм соберём от i-го процесса */
            recvdispls[i] = cum;                 /* смещение в результирующем массиве сумм */
            sendcounts[i] = (sums_count > 0) ? (sums_count + 1) : 0; /* сколько элементов отправить */
            displs[i] = cum;                     /* смещение в исходном массиве a для Scatterv */
            cum += recvcounts[i];
        }

        if (rank == 0) {
            printf("Iteration %d: curr_n=%d, M=%d, distribution:\n", iteration, curr_n, M);
            for (int i = 0; i < size; ++i) {
                printf("  proc %d: recvcounts=%d, sendcounts=%d, displ=%d, recvdispl=%d\n",
                        i, recvcounts[i], sendcounts[i], displs[i], recvdispls[i]);
            }
            printf("\n");
        }

        int local_count = sendcounts[rank]; /* число элементов, которые нам пришлёт root */
        if (local_count > 0) {
            local_a = (double*)realloc(local_a, (size_t)local_count * sizeof(double));
        } else {
            /* гарантируем валидный указатель для free */
            free(local_a);
            local_a = NULL;
        }

        MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE,
                     local_a, local_count, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        /* Показываем, что получил каждый процесс */
        printf("[proc %d] received %d elements: ", rank, local_count);
        for (int i = 0; i < local_count; ++i) printf("%.0f ", local_a ? local_a[i] : 0.0);
        printf("\n");

        int local_sums = (local_count > 0) ? (local_count - 1) : 0;
        if (local_sums > 0) {
            local_c = (double*)realloc(local_c, (size_t)local_sums * sizeof(double));
            for (int i = 0; i < local_sums; ++i) local_c[i] = local_a[i] + local_a[i + 1];
            printf("[proc %d] computed %d sums: ", rank, local_sums);
            for (int i = 0; i < local_sums; ++i) printf("%.0f ", local_c[i]);
            printf("\n");
        } else {
            free(local_c);
            local_c = NULL;
        }

        double *new_a = NULL;
        if (rank == 0) new_a = (double*)malloc((size_t)M * sizeof(double));

        /* Сбор результатов сумм на root */
        MPI_Gatherv(local_c, local_sums, MPI_DOUBLE,
                    new_a, recvcounts, recvdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            /* Здесь мы явно уменьшаем (shrink) массив: создаём новый массив new_a длины M
               и освобождаем старый a — это именно уменьшение длины массива на root в конце итерации. */
            printf("[root] after gathering (step %d): ", iteration);
            for (int i = 0; i < M; ++i) printf("%.0f ", new_a[i]);
            printf("\n\n");

            free(a); /* освобождаем старый массив */
            a = new_a; /* заменяем указатель — теперь a имеет длину M */
        } else {
            /* Не-root не хранит новый a */
            free(new_a);
        }

        /* Обновляем curr_n в конце итерации: длина стала M */
        curr_n = M;
    }

    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("Final result: %.0f\n", a[0]);
        printf("Total time: %f s\n", t1 - t0);
        free(a);
    }

    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
    if (local_a) free(local_a);
    if (local_c) free(local_c);

    MPI_Finalize();
    return 0;
}
