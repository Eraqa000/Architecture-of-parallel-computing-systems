#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10; // Using 10 elements for clearer demonstration

    /* Allocate / initialize on root */
    double *a = NULL;
    if (rank == 0) {
        a = (double*)malloc((size_t)N * sizeof(double));
        if (!a) {
            fprintf(stderr, "Allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; ++i) a[i] = (double)(i + 1);

        printf("Initial array (n=%d):\n", N);
        for (int i = 0; i < N; ++i) printf("%.0f ", a[i]);
        printf("\n\n");
    }

    /* Prepare helper arrays (counts in elements) */
    int *sendcounts = (int*)calloc(size, sizeof(int));
    int *displs = (int*)calloc(size, sizeof(int));
    int *recvcounts = (int*)calloc(size, sizeof(int));
    int *recvdispls = (int*)calloc(size, sizeof(int));

    double *local_a = NULL;
    double *local_c = NULL;

    int curr_n = N;
    double t0 = MPI_Wtime();
    int iteration = 0;

    while (curr_n > 1) {
        iteration++;
        int M = curr_n - 1; /* number of sums to compute */

        int base = M / size;
        int rem = M % size;
        int cum = 0;
        for (int i = 0; i < size; ++i) {
            int sums_count = base + (i < rem ? 1 : 0); /* how many sums process i computes */
            recvcounts[i] = sums_count;
            recvdispls[i] = cum;
            sendcounts[i] = (sums_count > 0) ? (sums_count + 1) : 0; /* elements */
            displs[i] = cum; /* element offset in a */
            cum += recvcounts[i];
        }

        // Debug output for distribution
        if (rank == 0) {
            printf("\nIteration %d - Distribution for n=%d:\n", iteration, curr_n);
            for (int i = 0; i < size; ++i) {
                printf("Process %d: handles %d elements starting at index %d\n", 
                       i, sendcounts[i], displs[i]);
            }
            printf("\n");
        }

        int local_count = sendcounts[rank]; /* number of elements to receive */
        if (local_count > 0) local_a = (double*)realloc(local_a, (size_t)local_count * sizeof(double));
        else local_a = (double*)realloc(local_a, sizeof(double));

        /* Scatterv with MPI_DOUBLE (counts are in elements) */
        MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE,
                     local_a, local_count, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        // Debug output for local data
        printf("Process %d received: ", rank);
        for (int i = 0; i < local_count; i++) {
            printf("%.0f ", local_a[i]);
        }
        printf("\n");

        int local_sums = (local_count > 0) ? (local_count - 1) : 0;
        if (local_sums > 0) local_c = (double*)realloc(local_c, (size_t)local_sums * sizeof(double));
        else local_c = (double*)realloc(local_c, sizeof(double));

        for (int i = 0; i < local_sums; ++i) {
            local_c[i] = local_a[i] + local_a[i + 1];
        }

        // Debug output for local sums
        if (local_sums > 0) {
            printf("Process %d computed sums: ", rank);
            for (int i = 0; i < local_sums; i++) {
                printf("%.0f ", local_c[i]);
            }
            printf("\n");
        }

        double *new_a = NULL;
        if (rank == 0) new_a = (double*)malloc((size_t)M * sizeof(double));

        /* Gatherv with MPI_DOUBLE (counts in elements) */
        MPI_Gatherv(local_c, local_sums, MPI_DOUBLE,
                    new_a, recvcounts, recvdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\nAfter step %d (n=%d -> n=%d):\n", iteration, curr_n, M);
            for (int i = 0; i < M; ++i) printf("%.0f ", new_a[i]);
            printf("\n\n");

            free(a);
            a = new_a;
        }

        /* Deterministic update of curr_n on all ranks */
        curr_n = M;
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("Final result: %.0f\n", a[0]);
        printf("Total time: %f seconds\n", t1 - t0);
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