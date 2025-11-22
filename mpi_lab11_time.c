#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    double *A = NULL, *B = NULL, *C = NULL;
    double *localA, *localC;
    int *sendcounts, *displs;
    int rows_per_proc, remainder, start_row;
    
    double start_time, end_time, compute_start_time, compute_end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Время старта программы
    if (rank == 0) {
        start_time = MPI_Wtime();
        printf("Enter matrix size N: ");
        fflush(stdout);
        scanf("%d", &N);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    sendcounts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    rows_per_proc = N / size;
    remainder = N % size;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    localA = malloc(sendcounts[rank] * sizeof(double));
    localC = malloc(sendcounts[rank] * sizeof(double));
    B = malloc(N * N * sizeof(double));

    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        srand((unsigned)time(NULL));

        for (int i = 0; i < N * N; i++)
            A[i] = rand() % 20;
        for (int i = 0; i < N * N; i++)
            B[i] = rand() % 20;

        printf("\nMatrix A:\n");
        

        printf("\nMatrix B:\n");
        
        printf("\n");
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 localA, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Время старта вычислений
    compute_start_time = MPI_Wtime();

    int local_rows = sendcounts[rank] / N;
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += localA[i * N + k] * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }

    if (rank == 0)
        C = malloc(N * N * sizeof(double));

    MPI_Gatherv(localC, sendcounts[rank], MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    compute_end_time = MPI_Wtime();

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("\nTotal time (including input/output): %f seconds\n", end_time - start_time);
        printf("Computation time: %f seconds\n", compute_end_time - compute_start_time);
    }

    MPI_Finalize();
    return 0;
}
