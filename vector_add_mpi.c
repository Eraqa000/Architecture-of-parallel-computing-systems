#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SIZEOF_INT128__)
typedef __int128 int128_t;
#else
#error "__int128 not supported by this compiler"
#endif

static void int128_to_str(int128_t v, char *buf, size_t bufsize) {
    if (bufsize == 0) return;
    char tmp[64];
    int pos = 0;
    int negative = 0;
    if (v == 0) {
        strncpy(buf, "0", bufsize);
        return;
    }
    if (v < 0) {
        negative = 1;
        v = -v;
    }
    while (v != 0 && pos < (int)(sizeof(tmp)-1)) {
        int digit = (int)(v % 10);
        tmp[pos++] = '0' + digit;
        v /= 10;
    }
    if (negative) tmp[pos++] = '-';
    int i;
    int len = (pos < (int)bufsize-1) ? pos : (int)bufsize-1;
    for (i = 0; i < len; i++) buf[i] = tmp[pos - 1 - i];
    buf[len] = '\0';
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Fixed input size (change here if needed)
    int N = 128; // <-- change N in code

    // Allocate / initialize on root
    int128_t *a = NULL;
    if (rank == 0) {
        a = (int128_t*)malloc((size_t)N * sizeof(int128_t));
        if (!a) {
            fprintf(stderr, "Allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; ++i) a[i] = (int128_t)(i + 1);

        printf("Initial array (n=%d):\n", N);
        for (int i = 0; i < N; ++i) {
            char buf[64];
            int128_to_str(a[i], buf, sizeof(buf));
            printf("%s ", buf);
        }
        printf("\n\n");
    }

    // Prepare helper arrays (counts in elements and in bytes)
    int *sendcounts_elems = (int*)calloc(size, sizeof(int));
    int *displs_elems = (int*)calloc(size, sizeof(int));
    int *recvcounts_elems = (int*)calloc(size, sizeof(int));
    int *recvdispls_elems = (int*)calloc(size, sizeof(int));

    int *sendcounts_bytes = (int*)calloc(size, sizeof(int));
    int *displs_bytes = (int*)calloc(size, sizeof(int));
    int *recvcounts_bytes = (int*)calloc(size, sizeof(int));
    int *recvdispls_bytes = (int*)calloc(size, sizeof(int));

    int128_t *local_a = NULL;
    int128_t *local_c = NULL;

    int curr_n = N;

    double t0 = MPI_Wtime();
    int iteration = 0;

    while (curr_n > 1) {
        iteration++;
        int M = curr_n - 1; // number of sums to compute

        int base = M / size;
        int rem = M % size;
        int cum = 0;
        for (int i = 0; i < size; ++i) {
            int sums_count = base + (i < rem ? 1 : 0); // how many sums process i computes
            recvcounts_elems[i] = sums_count;
            recvdispls_elems[i] = cum;
            sendcounts_elems[i] = (sums_count > 0) ? (sums_count + 1) : 0; // elements
            displs_elems[i] = cum; // element offset in a
            cum += recvcounts_elems[i];
        }

        // Convert element counts to byte counts for MPI_BYTE operations
        for (int i = 0; i < size; ++i) {
            sendcounts_bytes[i] = sendcounts_elems[i] * (int)sizeof(int128_t);
            displs_bytes[i] = displs_elems[i] * (int)sizeof(int128_t);
            recvcounts_bytes[i] = recvcounts_elems[i] * (int)sizeof(int128_t);
            recvdispls_bytes[i] = recvdispls_elems[i] * (int)sizeof(int128_t);
        }

        int local_count = sendcounts_elems[rank]; // number of int128_t elements to receive
        if (local_count > 0) local_a = (int128_t*)realloc(local_a, (size_t)local_count * sizeof(int128_t));
        else local_a = (int128_t*)realloc(local_a, sizeof(int128_t));

        // Scatterv as bytes
        MPI_Scatterv(a, sendcounts_bytes, displs_bytes, MPI_BYTE,
                     local_a, local_count * (int)sizeof(int128_t), MPI_BYTE,
                     0, MPI_COMM_WORLD);

        int local_sums = (local_count > 0) ? (local_count - 1) : 0;
        if (local_sums > 0) local_c = (int128_t*)realloc(local_c, (size_t)local_sums * sizeof(int128_t));
        else local_c = (int128_t*)realloc(local_c, sizeof(int128_t));

        for (int i = 0; i < local_sums; ++i) {
            local_c[i] = local_a[i] + local_a[i + 1];
        }

        int128_t *new_a = NULL;
        if (rank == 0) new_a = (int128_t*)malloc((size_t)M * sizeof(int128_t));

        // Gatherv as bytes
        MPI_Gatherv(local_c, local_sums * (int)sizeof(int128_t), MPI_BYTE,
                    new_a, recvcounts_bytes, recvdispls_bytes, MPI_BYTE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("After step (n=%d -> n=%d):\n", curr_n, M);
            for (int i = 0; i < M; ++i) {
                char buf[64];
                int128_to_str(new_a[i], buf, sizeof(buf));
                printf("%s ", buf);
            }
            printf("\n\n");

            free(a);
            a = new_a;
        }

        // Deterministic update of curr_n on all ranks
        curr_n = M;
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        char buf[64];
        int128_to_str(a[0], buf, sizeof(buf));
        printf("Final result: %s\n", buf);
        printf("Total time: %f seconds\n", t1 - t0);
        free(a);
    }

    free(sendcounts_elems);
    free(displs_elems);
    free(recvcounts_elems);
    free(recvdispls_elems);
    free(sendcounts_bytes);
    free(displs_bytes);
    free(recvcounts_bytes);
    free(recvdispls_bytes);
    if (local_a) free(local_a);
    if (local_c) free(local_c);

    MPI_Finalize();
    return 0;
}
