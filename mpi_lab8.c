#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

long long safe_add(long long a, long long b) {
    if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b)) {
        printf("Warning: Overflow detected! Results may be incorrect.\n");
        return a; 
    }
    return a + b;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int i;

    double start_time, end_time;      
    double iter_start, iter_end;     
    int iteration = 0;                

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();

    int N = 64; 


    int curr_n = N;
    long long *a = NULL;
    if (rank == 0) {
        a = (long long*)malloc(curr_n * sizeof(long long));
        for (i = 0; i < curr_n; i++) a[i] = i + 1;

        printf("Initial array (n=%d):\n", curr_n);
        for (i = 0; i < curr_n; i++) printf("%d ", a[i]);
        printf("\n\n");
    }

    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    int *recvcounts = (int*)malloc(size * sizeof(int));
    int *recvdispls = (int*)malloc(size * sizeof(int));
    long long *local_a = NULL;
    long long *local_c = NULL;

    while (curr_n > 1) {
        iter_start = MPI_Wtime();
        iteration++;
        int M = curr_n - 1; 

        int base = M / size;
        int rem = M % size;
        int cum = 0;
        for (i = 0; i < size; i++) {
            int sums_count = base + (i < rem ? 1 : 0);
            recvcounts[i] = sums_count; 
            recvdispls[i] = cum; 
            sendcounts[i] = (sums_count > 0) ? (sums_count + 1) : 0;
            displs[i] = cum; 
            cum += recvcounts[i];
        }

        int local_count = sendcounts[rank];
        if (local_count > 0) {
            local_a = (long long*)realloc(local_a, local_count * sizeof(long long));
        } else {
            local_a = (long long*)realloc(local_a, sizeof(long long));
        }

    MPI_Scatterv(a, sendcounts, displs, MPI_LONG_LONG,
             local_a, local_count, MPI_LONG_LONG,
             0, MPI_COMM_WORLD);

        int local_sums = (local_count > 0) ? (local_count - 1) : 0;
        if (local_sums > 0) {
            local_c = (long long*)realloc(local_c, local_sums * sizeof(long long));
            for (i = 0; i < local_sums; i++) local_c[i] = safe_add(local_a[i], local_a[i + 1]);
        } else {
            local_c = (long long*)realloc(local_c, sizeof(long long));
        }

        long long *new_a = NULL;
        if (rank == 0) {
            new_a = (long long*)malloc(M * sizeof(long long));
        }

    MPI_Gatherv(local_c, local_sums, MPI_LONG_LONG,
            new_a, recvcounts, recvdispls, MPI_LONG_LONG,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("After step (n=%d -> n=%d):\n", curr_n, M);
            for (i = 0; i < M; i++) printf("%lld ", new_a[i]);
            iter_end = MPI_Wtime();
            printf("\nIteration %d time: %f seconds\n\n", iteration, iter_end - iter_start);

            // free(a);
            a = new_a;
        }

        // Все процессы локально обновляют curr_n детерминировано
        curr_n = M;
    }

    // Конец цикла
    end_time = MPI_Wtime();

    if (rank == 0) {
        if (a != NULL) {
            printf("Final result: %lld\n", a[0]);
            printf("\nPerformance summary:\n");
            printf("Total execution time: %f seconds\n", end_time - start_time);
            printf("Number of processes: %d\n", size);
            printf("Input size: %d\n", N);
            free(a);
        }
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