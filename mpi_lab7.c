#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fflush(stdout);

    // ---- 2 процесс арасындағы алмасу ----
    if (size >= 2) {
        if (rank == 0) {
            int number = 42;
            printf("Process %d -> Process 1: sent number = %d\n", rank, number);
            fflush(stdout);

            MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&number, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("Process %d: received number = %d\n", rank, number);
            fflush(stdout);
        } else if (rank == 1) {
            int received;
            MPI_Recv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d: received number = %d\n", rank, received);
            fflush(stdout);

            received += 10;
            MPI_Send(&received, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

            printf("Process %d -> Process 0: reply sent.\n", rank);
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- 4 процесс арасындағы алмасу ----
    if (size >= 4) {
        if (rank == 0) {
            int value = 99;
            for (int i = 1; i < size; i++) {
                MPI_Send(&value, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
                printf("Process 0 -> Process %d: message sent (tag=100)\n", i);
                fflush(stdout);
            }

            for (int i = 1; i < size; i++) {
                int resp;
                MPI_Recv(&resp, 1, MPI_INT, i, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Process 0: received reply from process %d (tag=200)\n", resp);
                fflush(stdout);
            }
        } else {
            int recv_val;
            MPI_Recv(&recv_val, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d: received message = %d\n", rank, recv_val);
            fflush(stdout);

            MPI_Send(&rank, 1, MPI_INT, 0, 200, MPI_COMM_WORLD);
            printf("Process %d -> Process 0: reply sent (tag=200)\n", rank);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}

