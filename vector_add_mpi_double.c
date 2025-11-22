#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 64; 

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
        int M = curr_n - 1; //  бұл жерде біздегі массивтен қанша қосындылар саны шығатыны көрсетіледі

        int base = M / size;  // М-ды проц санына бөлгенде әр процқа қанша қосындыдан келеді
        int rem = M % size;   // қанша алғашқы процқа бір қосымша қосындыдан келетінін анықтау үшін керек
        int cum = 0;  // коммулятивный счетчик recvdispls, displs үшін жылжуды есептеуге арналған
        for (int i = 0; i < size; ++i) { // әр процессорға беру
            int sums_count = base + (i < rem ? 1 : 0); // қай процессор қанша қосынды есептейтінін анықтаймыз
            recvcounts[i] = sums_count;  // root қа i-ші процессордан қанша қосынды жинау керек екенін көрсетуге арналған
            recvdispls[i] = cum; // root та қосындыларды қай позициядан бастап сақтау керектігін көрсетуге арналған
            sendcounts[i] = (sums_count > 0) ? (sums_count + 1) : 0; // қанша элемент жіберу керектігін көрсетеміз
            displs[i] = cum; // а дан қай индекстен бастап жіберу керектігін көрсетеміз
            cum += recvcounts[i]; // жаңартылады
        }

        int local_count = sendcounts[rank];  // әр процесс өзіне қанша эл келетінін анықтайды
        if (local_count > 0) local_a = (double*)realloc(local_a, (size_t)local_count * sizeof(double));
        else local_a = (double*)realloc(local_a, sizeof(double));

        MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE,
                     local_a, local_count, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        int local_sums = (local_count > 0) ? (local_count - 1) : 0; // канша косындылар болатыны 
        if (local_sums > 0) local_c = (double*)realloc(local_c, (size_t)local_sums * sizeof(double));
        else local_c = (double*)realloc(local_c, sizeof(double));

        for (int i = 0; i < local_sums; ++i) {
            local_c[i] = local_a[i] + local_a[i + 1];
        }

        double *new_a = NULL;
        if (rank == 0) new_a = (double*)malloc((size_t)M * sizeof(double));

        MPI_Gatherv(local_c, local_sums, MPI_DOUBLE,
                    new_a, recvcounts, recvdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("After step (n=%d -> n=%d):\n", curr_n, M);
            for (int i = 0; i < M; ++i) printf("%.0f ", new_a[i]);
            printf("\n\n");

            free(a);
            a = new_a;
        }

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
