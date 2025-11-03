#define _POSIX_C_SOURCE 200809L
#include <sys/types.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* allocate zeroed matrix (rows x cols) */
static double *calloc_matrix(size_t rows, size_t cols) {
    double *m = calloc(rows * cols, sizeof(double));
    if (!m) {
        fprintf(stderr, "calloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return m;
}

/* -------------------- Matrix reading from file (use streaming for very large files >500MB) -------------------- */
void read_matrix_streaming_file(const char *filename, double **A_out, double **B_out, int *m_out, int *k_out, int *n_out) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int readingA = 0, readingB = 0;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

    // First pass - determine dimensions
    while ((read = getline(&line, &len, fp)) != -1) {
        if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
        if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
        if (!readingA && !readingB) continue;

        int count = 0;
        char *tmp = strtok(line, " \t\n");
        while (tmp) { count++; tmp = strtok(NULL, " \t\n"); }
        if (count == 0) continue;

        if (readingA) { rowsA++; colsA = count; }
        else if (readingB) { rowsB++; colsB = count; }
    }

    if (rowsA == 0 || rowsB == 0) {
        fprintf(stderr, "Error: input file must contain Matrix A and Matrix B\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    rewind(fp);
    double *A = malloc((size_t)rowsA * colsA * sizeof(double));
    double *B = malloc((size_t)rowsB * colsB * sizeof(double));
    if (!A || !B) {
        fprintf(stderr, "malloc failed for A or B\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    readingA = readingB = 0;
    int idxA = 0, idxB = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
        if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
        if (!readingA && !readingB) continue;

        char *tmp = strtok(line, " \t\n");
        while (tmp) {
            double val = atof(tmp);
            if (readingA) A[idxA++] = val;
            else if (readingB) B[idxB++] = val;
            tmp = strtok(NULL, " \t\n");
        }
    }

    free(line);
    fclose(fp);

    *A_out = A;
    *B_out = B;
    *m_out = rowsA;
    *k_out = colsA;
    *n_out = colsB;
}

/* -------------------- Write matrix (top-left m x n of full matrix with stride) -------------------- */
void write_matrix_to_file(const char *filename, double *Cfull, int m, int n, int stride) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open output file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%.4f", Cfull[i * stride + j]);
            if (j < n - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* -------------------- Local optimized multiply (n x n) -------------------- */
void local_matmul(int n, double *A, double *B, double *C) {
    /* i-k-j ordering for better cache reuse */
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

/* -------------------- Pack global N x N matrix into packed blocks (rank order 0..p-1) ----------
   Each block is block_size x block_size stored contiguously in row-major inside packed:
   packed[rank * block_elems + i*block_size + j] = global[(bi*block_size + i)*N + (bj*block_size + j)]
*/
void pack_global_to_packed(double *global, double *packed, int N, int q, int block_size) {
    int block_elems = block_size * block_size;
    for (int bi = 0; bi < q; bi++) {
        for (int bj = 0; bj < q; bj++) {
            int rank = bi * q + bj;
            double *dst = packed + (size_t)rank * block_elems;
            for (int i = 0; i < block_size; i++) {
                int global_row = bi * block_size + i;
                int global_col = bj * block_size;
                const double *src = global + (size_t)global_row * N + global_col;
                memcpy(dst + (size_t)i * block_size, src, (size_t)block_size * sizeof(double));
            }
        }
    }
}

/* -------------------- Unpack packed blocks into global N x N matrix -------------------- */
void unpack_packed_to_global(double *packed, double *global, int N, int q, int block_size) {
    int block_elems = block_size * block_size;
    for (int bi = 0; bi < q; bi++) {
        for (int bj = 0; bj < q; bj++) {
            int rank = bi * q + bj;
            double *src = packed + (size_t)rank * block_elems;
            for (int i = 0; i < block_size; i++) {
                int global_row = bi * block_size + i;
                int global_col = bj * block_size;
                double *dst = global + (size_t)global_row * N + global_col;
                memcpy(dst, src + (size_t)i * block_size, (size_t)block_size * sizeof(double));
            }
        }
    }
}

/* -------------------- Cannon multiply on local blocks --------------------
   local_A, local_B, local_C are block_size x block_size contiguous buffers.
   cart_comm is the 2D Cartesian communicator with periods = {1,1}.
*/
void cannon_local(int Npad, int q, int block_size,
                  double *local_A, double *local_B, double *local_C,
                  MPI_Comm cart_comm) {

    int npes, rank;
    MPI_Comm_size(cart_comm, &npes);
    MPI_Comm_rank(cart_comm, &rank);

    int dims[2], periods[2], coords[2];
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);
    /* neighbors: left/right along column dimension (1), up/down along row dimension (0) */
    int leftrank, rightrank, uprank, downrank;
    MPI_Cart_shift(cart_comm, 1, +1, &leftrank, &rightrank); // shift along columns
    MPI_Cart_shift(cart_comm, 0, +1, &uprank, &downrank);   // shift along rows

    MPI_Status status;

    /* Initial alignment (shift A left by coords[0], B up by coords[1]) */
    /* Use Cart_shift with negative displacements as in original starter logic */
    int shiftsource, shiftdest;
    MPI_Cart_shift(cart_comm, 1, -coords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE,
                         shiftdest, 100, shiftsource, 100, cart_comm, &status);
    MPI_Cart_shift(cart_comm, 0, -coords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE,
                         shiftdest, 200, shiftsource, 200, cart_comm, &status);

    /* Main loop: q steps */
    for (int step = 0; step < q; step++) {
        local_matmul(block_size, local_A, local_B, local_C);

        /* Rotate A left (send to left, recv from right) */
        MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE,
                             leftrank, 1000 + step, rightrank, 1000 + step, cart_comm, &status);
        /* Rotate B up (send to up, recv from down) */
        MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE,
                             uprank, 2000 + step, downrank, 2000 + step, cart_comm, &status);
    }

    /* (optionally) restore original distribution - not required before gather, skip to save time */
}

/* -------------------- Main program -------------------- */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *input_file = argv[1];
    const char *output_file = argv[2];

    /* Read A and B on rank 0 (streaming) */
    double *A = NULL, *B = NULL;
    int m = 0, k = 0, n = 0;
    if (rank == 0) {
        read_matrix_streaming_file(input_file, &A, &B, &m, &k, &n);
        /* verify inner dims */
        if (k <= 0) {
            fprintf(stderr, "invalid k\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast dimensions to all ranks */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Validate dims locally */
    if (m <= 0 || k <= 0 || n <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid matrix dimensions\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Determine pgrid (q) */
    int q = (int)round(sqrt((double)npes));
    if (q * q != npes) {
        if (rank == 0) fprintf(stderr, "Number of processes must be a perfect square (1,4,9,...)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Determine padded square dimension Npad = next multiple of q >= max(m,k,n) */
    int S = m;
    if (k > S) S = k;
    if (n > S) S = n;
    int Npad = ((S + q - 1) / q) * q; /* multiple of q */

    /* On rank 0 build padded Apad (Npad x Npad) and Bpad (Npad x Npad) */
    double *Apad = NULL, *Bpad = NULL;
    if (rank == 0) {
        Apad = calloc_matrix(Npad, Npad);
        Bpad = calloc_matrix(Npad, Npad);
        /* copy A (m x k) into top-left of Apad */
        for (int i = 0; i < m; i++) {
            memcpy(Apad + (size_t)i * Npad, A + (size_t)i * k, (size_t)k * sizeof(double));
        }
        /* copy B (k x n) into top-left of Bpad */
        for (int i = 0; i < k; i++) {
            memcpy(Bpad + (size_t)i * Npad, B + (size_t)i * n, (size_t)n * sizeof(double));
        }
    }

    /* Determine block size and block elems */
    int block_size = Npad / q;
    if (block_size <= 0) {
        fprintf(stderr, "block_size <= 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int block_elems = block_size * block_size;

    /* Prepare packed send buffers on root (one contiguous block per rank) */
    double *packedA = NULL;
    double *packedB = NULL;
    if (rank == 0) {
        packedA = malloc((size_t)npes * block_elems * sizeof(double));
        packedB = malloc((size_t)npes * block_elems * sizeof(double));
        if (!packedA || !packedB) {
            fprintf(stderr, "malloc packed buffers failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        pack_global_to_packed(Apad, packedA, Npad, q, block_size);
        pack_global_to_packed(Bpad, packedB, Npad, q, block_size);
    }

    /* Each rank allocates local blocks once */
    double *local_A = calloc_matrix(block_size, block_size);
    double *local_B = calloc_matrix(block_size, block_size);
    double *local_C = calloc_matrix(block_size, block_size);

    /* Scatter packed blocks using MPI_Scatter (contiguous chunks) */
    MPI_Scatter(packedA, block_elems, MPI_DOUBLE,
                local_A, block_elems, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Scatter(packedB, block_elems, MPI_DOUBLE,
                local_B, block_elems, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* root can free packed buffers and Apad/Bpad/A/B now */
    if (rank == 0) {
        free(packedA); packedA = NULL;
        free(packedB); packedB = NULL;
        free(Apad); Apad = NULL;
        free(Bpad); Bpad = NULL;
        free(A); A = NULL;
        free(B); B = NULL;
    }

    /* Create cartesian communicator for Cannon (periodic) */
    int dims[2] = { q, q };
    int periods[2] = { 1, 1 };
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    /* Synchronize and time only the Cannon multiplication part */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Run Cannon's algorithm on local blocks */
    cannon_local(Npad, q, block_size, local_A, local_B, local_C, cart_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("Time for matrix multiplication: %.6f seconds\n", t1 - t0);
    }

    /* Gather local_C blocks into packedC on root (contiguous) */
    double *packedC = NULL;
    if (rank == 0) {
        packedC = malloc((size_t)npes * block_elems * sizeof(double));
        if (!packedC) {
            fprintf(stderr, "malloc packedC failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(local_C, block_elems, MPI_DOUBLE,
               packedC, block_elems, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /* Unpack packedC into Cpad (Npad x Npad) on root, then write top-left m x n */
    if (rank == 0) {
        double *Cpad = calloc_matrix(Npad, Npad);
        unpack_packed_to_global(packedC, Cpad, Npad, q, block_size);

        write_matrix_to_file(output_file, Cpad, m, n, Npad);

        free(Cpad);
        free(packedC);
    }

    /* Cleanup local buffers and communicator */
    free(local_A); free(local_B); free(local_C);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}

// Part A
// #define _POSIX_C_SOURCE 200809L  // enables getline() and ssize_t on most compilers
// #include <sys/types.h>
// #include <mpi.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>

// /* ---------- Matrix reading from file (streaming) ---------- */
// /* Now records rowsA, colsA, rowsB, colsB (does not assume square) */
// void read_matrix_streaming(const char *filename,
//                            double **A, double **B,
//                            int *rowsA, int *colsA, int *rowsB, int *colsB) {
//     FILE *fp = fopen(filename, "r");
//     if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

//     char *line = NULL;
//     size_t len = 0;
//     ssize_t read;
//     int readingA = 0, readingB = 0;

//     int rA = 0, cA = 0, rB = 0, cB = 0;

//     // First pass — determine dims
//     while ((read = getline(&line, &len, fp)) != -1) {
//         if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
//         if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
//         if (!readingA && !readingB) continue;

//         int count = 0;
//         char *tmp = strtok(line, " \t\n");
//         while (tmp) { count++; tmp = strtok(NULL, " \t\n"); }
//         if (count == 0) continue;

//         if (readingA) { rA++; cA = count; }
//         else if (readingB) { rB++; cB = count; }
//     }

//     if (rA == 0 || rB == 0) {
//         fprintf(stderr, "Error: empty matrices or wrong file format\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     rewind(fp);

//     *rowsA = rA; *colsA = cA; *rowsB = rB; *colsB = cB;

//     if (cA != rB) {
//         fprintf(stderr, "Error: inner dimensions do not match (colsA=%d != rowsB=%d)\n", cA, rB);
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     *A = malloc((size_t)rA * cA * sizeof(double));
//     *B = malloc((size_t)rB * cB * sizeof(double));
//     if (!*A || !*B) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

//     int idxA = 0, idxB = 0;
//     readingA = readingB = 0;

//     // Second pass — parse
//     while ((read = getline(&line, &len, fp)) != -1) {
//         if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
//         if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
//         if (!readingA && !readingB) continue;

//         char *tmp = strtok(line, " \t\n");
//         while (tmp) {
//             double val = atof(tmp);
//             if (readingA) (*A)[idxA++] = val;
//             else if (readingB) (*B)[idxB++] = val;
//             tmp = strtok(NULL, " \t\n");
//         }
//     }

//     free(line);
//     fclose(fp);
// }

// /* ---------- Local matrix multiplication for rectangular blocks ----------
//    A: m_local x k_local
//    B: k_local x n_local
//    C: m_local x n_local (accumulates)
// */
// void MatrixMultiply_rect(int m_local, int k_local, int n_local,
//                          double *A, double *B, double *C) {
//     for (int i = 0; i < m_local; i++) {
//         for (int j = 0; j < n_local; j++) {
//             double sum = 0.0;
//             for (int k = 0; k < k_local; k++) {
//                 sum += A[i * k_local + k] * B[k * n_local + j];
//             }
//             C[i * n_local + j] += sum;
//         }
//     }
// }

// /* ---------- Write matrix to output file (m x n) ---------- */
// void write_matrix_to_file_rect(const char *filename, double *C, int m, int n) {
//     FILE *fp = fopen(filename, "w");
//     if (!fp) {
//         fprintf(stderr, "Error: Cannot open output file %s\n", filename);
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             fprintf(fp, "%.4f", C[i * n + j]);
//             if (j < n - 1) fprintf(fp, " ");
//         }
//         fprintf(fp, "\n");
//     }
//     fclose(fp);
// }

// /* ---------- Create Cartesian grid communicator ---------- */
// /* Returns grid and sets p_rows/p_cols via dims in MPI_Cart_get when needed */
// MPI_Comm create_cartesian_grid(MPI_Comm base_comm, int *p_rows_out, int *p_cols_out) {
//     int npes;
//     MPI_Comm_size(base_comm, &npes);

//     int dims[2] = {0, 0};
//     MPI_Dims_create(npes, 2, dims); // fill dims so that dims[0]*dims[1] = npes

//     int periods[2] = {1, 1};  /* wraparound for Cannon */
//     MPI_Comm grid;
//     MPI_Cart_create(base_comm, 2, dims, periods, 1, &grid);

//     if (p_rows_out || p_cols_out) {
//         int dims_out[2], coords_dummy[2];
//         MPI_Cart_get(grid, 2, dims_out, periods, coords_dummy);
//         if (p_rows_out) *p_rows_out = dims_out[0];
//         if (p_cols_out) *p_cols_out = dims_out[1];
//     }

//     return grid;
// }

// /* ---------- Scatter a general 2D (rows x cols) global matrix into blocks ----------
//    Each block has size (rows/p_rows) x (cols/p_cols). Works for non-square global dims.
//    local_buf must be preallocated with block_rows * block_cols doubles.
//    Uses grid communicator.
// */
// void scatter_2d_blocks_general(double *global, double *local,
//                                int rows, int cols,
//                                int p_rows, int p_cols,
//                                MPI_Comm grid, int tag) {
//     int npes, rank;
//     MPI_Comm_size(grid, &npes);
//     MPI_Comm_rank(grid, &rank);

//     int dims[2], periods[2], coords[2];
//     MPI_Cart_get(grid, 2, dims, periods, coords);
//     int block_rows = rows / p_rows;
//     int block_cols = cols / p_cols;

//     if (rank == 0) {
//         for (int r = 0; r < npes; r++) {
//             int rc[2];
//             MPI_Cart_coords(grid, r, 2, rc);

//             int sizes[2]    = { rows, cols };
//             int subsizes[2] = { block_rows, block_cols };
//             int starts[2]   = { rc[0] * block_rows, rc[1] * block_cols };

//             MPI_Datatype sub;
//             MPI_Type_create_subarray(2, sizes, subsizes, starts,
//                                      MPI_ORDER_C, MPI_DOUBLE, &sub);
//             MPI_Type_commit(&sub);

//             if (r == 0) {
//                 // copy into local directly
//                 for (int i = 0; i < block_rows; i++) {
//                     const double *src = global + (starts[0] + i) * cols + starts[1];
//                     double *dst = local + i * block_cols;
//                     memcpy(dst, src, block_cols * sizeof(double));
//                 }
//             } else {
//                 MPI_Send(global, 1, sub, r, tag, grid);
//             }
//             MPI_Type_free(&sub);
//         }
//     } else {
//         MPI_Recv(local, block_rows * block_cols, MPI_DOUBLE, 0, tag, grid, MPI_STATUS_IGNORE);
//     }
// }

// /* ---------- Gather blocks back into general 2D global matrix ----------
//    local blocks are block_rows x block_cols; global is rows x cols.
// */
// void gather_2d_blocks_general(double *local, double *global,
//                               int rows, int cols,
//                               int p_rows, int p_cols,
//                               MPI_Comm grid, int tag) {
//     int npes, rank;
//     MPI_Comm_size(grid, &npes);
//     MPI_Comm_rank(grid, &rank);

//     int dims[2], periods[2], coords[2];
//     MPI_Cart_get(grid, 2, dims, periods, coords);
//     int block_rows = rows / p_rows;
//     int block_cols = cols / p_cols;

//     if (rank == 0) {
//         for (int r = 0; r < npes; r++) {
//             int rc[2];
//             MPI_Cart_coords(grid, r, 2, rc);

//             int sizes[2]    = { rows, cols };
//             int subsizes[2] = { block_rows, block_cols };
//             int starts[2]   = { rc[0] * block_rows, rc[1] * block_cols };

//             MPI_Datatype sub;
//             MPI_Type_create_subarray(2, sizes, subsizes, starts,
//                                      MPI_ORDER_C, MPI_DOUBLE, &sub);
//             MPI_Type_commit(&sub);

//             if (r == 0) {
//                 for (int i = 0; i < block_rows; i++) {
//                     double *dst = global + (starts[0] + i) * cols + starts[1];
//                     const double *src = local + i * block_cols;
//                     memcpy(dst, src, block_cols * sizeof(double));
//                 }
//             } else {
//                 MPI_Recv(global, 1, sub, r, tag, grid, MPI_STATUS_IGNORE);
//             }
//             MPI_Type_free(&sub);
//         }
//     } else {
//         MPI_Send(local, block_rows * block_cols, MPI_DOUBLE, 0, tag, grid);
//     }
// }

// /* ---------- Cannon's algorithm generalized for rectangular local blocks ----------
//    A_local: m_local x k_local
//    B_local: k_local x n_local
//    C_local: m_local x n_local
// */
// void MatrixMatrixMultiply_general(int p_rows, int p_cols,
//                                   int m_local, int k_local, int n_local,
//                                   double *A_local, double *B_local, double *C_local,
//                                   MPI_Comm grid) {
//     int npes, myrank, mycoords[2];
//     MPI_Comm_size(grid, &npes);
//     MPI_Comm_rank(grid, &myrank);
//     int periods[2], dims[2];
//     MPI_Cart_get(grid, 2, dims, periods, mycoords);

//     MPI_Status status;

//     /* Determine single-step neighbors for shifting by 1 */
//     int left, right, up, down;
//     MPI_Cart_shift(grid, 1, 1, &left, &right); // direction=1 (columns)
//     MPI_Cart_shift(grid, 0, 1, &up, &down);    // direction=0 (rows)

//     /* ---------- Initial alignment (skewing) ----------
//        Shift A left by mycoords[0] (row index)
//        Shift B up   by mycoords[1] (col index)
//     */
//     // shift A left by mycoords[0] positions: use MPI_Cart_shift with disp = mycoords[0]
//     int A_src, A_dst;
//     MPI_Cart_shift(grid, 1, mycoords[0], &A_src, &A_dst);
//     // A_src = coordinate col - disp (left-most source), A_dst = col + disp (right-most)
//     // To shift left by disp: send to left (A_src), receive from right (A_dst)
//     MPI_Sendrecv_replace(A_local, m_local * k_local, MPI_DOUBLE,
//                          A_src, 100,  // dest = left
//                          A_dst, 100,  // source = right
//                          grid, &status);

//     // shift B up by mycoords[1] positions: displacement = mycoords[1]
//     int B_src, B_dst;
//     MPI_Cart_shift(grid, 0, mycoords[1], &B_src, &B_dst);
//     // To shift up by disp: send to up (B_src), receive from down (B_dst)
//     MPI_Sendrecv_replace(B_local, k_local * n_local, MPI_DOUBLE,
//                          B_src, 200,
//                          B_dst, 200,
//                          grid, &status);

//     /* ---------- Main Cannon loop ----------
//        Repeat p_rows (== p_cols) times
//     */
//     int p = dims[0]; // dims[0] == dims[1] in our tests (square process grid)
//     for (int iter = 0; iter < p; iter++) {
//         // Multiply local blocks
//         MatrixMultiply_rect(m_local, k_local, n_local, A_local, B_local, C_local);

//         // Rotate A left by 1: send to left neighbor, receive from right neighbor
//         MPI_Sendrecv_replace(A_local, m_local * k_local, MPI_DOUBLE,
//                              left, 1000 + iter,
//                              right, 1000 + iter,
//                              grid, &status);

//         // Rotate B up by 1: send to up neighbor, receive from down neighbor
//         MPI_Sendrecv_replace(B_local, k_local * n_local, MPI_DOUBLE,
//                              up, 2000 + iter,
//                              down, 2000 + iter,
//                              grid, &status);
//     }

//     /* ---------- Restore original distribution (undo initial skew) ----------
//        Shift A right by mycoords[0] and B down by mycoords[1]
//     */
//     // For A: use MPI_Cart_shift with disp = mycoords[0] to get neighbors
//     MPI_Cart_shift(grid, 1, mycoords[0], &A_src, &A_dst);
//     // To shift right by disp: send to right (A_dst), receive from left (A_src)
//     MPI_Sendrecv_replace(A_local, m_local * k_local, MPI_DOUBLE,
//                          A_dst, 300,
//                          A_src, 300,
//                          grid, &status);

//     // For B: use MPI_Cart_shift with disp = mycoords[1]
//     MPI_Cart_shift(grid, 0, mycoords[1], &B_src, &B_dst);
//     // To shift down by disp: send to down (B_dst), receive from up (B_src)
//     MPI_Sendrecv_replace(B_local, k_local * n_local, MPI_DOUBLE,
//                          B_dst, 400,
//                          B_src, 400,
//                          grid, &status);
// }

// /* ---------- Main ---------- */
// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     int rank, npes;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &npes);

//     if (argc < 3) {
//         if (rank == 0)
//             fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
//         MPI_Finalize();
//         return 1;
//     }
//     const char *input_file = argv[1];
//     const char *output_file = argv[2];

//     double *A = NULL, *B = NULL, *C_full = NULL;
//     int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

//     if (rank == 0) {
//         read_matrix_streaming(input_file, &A, &B, &rowsA, &colsA, &rowsB, &colsB);
//         // we already validated colsA == rowsB in read function
//     }

//     // Broadcast dimensions to all ranks
//     int dims_buf[4];
//     if (rank == 0) {
//         dims_buf[0] = rowsA; dims_buf[1] = colsA; dims_buf[2] = rowsB; dims_buf[3] = colsB;
//     }
//     MPI_Bcast(dims_buf, 4, MPI_INT, 0, MPI_COMM_WORLD);
//     if (rank != 0) {
//         rowsA = dims_buf[0];
//         colsA = dims_buf[1];
//         rowsB = dims_buf[2];
//         colsB = dims_buf[3];
//     }

//     // Create cartesian grid and get process grid dims
//     int p_rows = 0, p_cols = 0;
//     MPI_Comm grid = create_cartesian_grid(MPI_COMM_WORLD, &p_rows, &p_cols);

//     // Check that process grid is square (problem statement uses sqrt(P) x sqrt(P))
//     if (p_rows != p_cols) {
//         if (rank == 0) fprintf(stderr, "Error: process grid must be square (p_rows == p_cols)\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     int p = p_rows;
//     // Validate divisibility assumption for part (a): each matrix dimension divisible by p
//     if ((rowsA % p) != 0 || (colsA % p) != 0 || (colsB % p) != 0) {
//         if (rank == 0) {
//             fprintf(stderr,
//                     "Error: In this part (a) we require rowsA, colsA, colsB to be divisible by sqrt(P) (p=%d)\n"
//                     "rowsA=%d colsA=%d colsB=%d\n",
//                     p, rowsA, colsA, colsB);
//         }
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     // compute local block sizes:
//     // A (rowsA x colsA) -> split into p_rows x p_cols blocks => local A: m_local x k_local
//     int m_local = rowsA / p_rows;
//     int k_local = colsA / p_cols;
//     // B (rowsB x colsB) where rowsB == colsA
//     int k_local_B = rowsB / p_rows; // should equal k_local (but check)
//     if (k_local_B != k_local) {
//         if (rank == 0) fprintf(stderr, "Internal error: inconsistent local k sizes\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }
//     int n_local = colsB / p_cols;
//     // C local size: m_local x n_local

//     // Allocate local buffers
//     double *A_local = malloc((size_t)m_local * k_local * sizeof(double));
//     double *B_local = malloc((size_t)k_local * n_local * sizeof(double));
//     double *C_local = calloc((size_t)m_local * n_local, sizeof(double));
//     if (!A_local || !B_local || !C_local) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

//     // Proper 2D scatter for A and B
//     // Use different tags to avoid confusion
//     scatter_2d_blocks_general(A, A_local, rowsA, colsA, p_rows, p_cols, grid, 1111);
//     scatter_2d_blocks_general(B, B_local, rowsB, colsB, p_rows, p_cols, grid, 2222);

//     // run Cannon's algorithm (generalized)
//     double start = MPI_Wtime();
//     MatrixMatrixMultiply_general(p_rows, p_cols, m_local, k_local, n_local,
//                                  A_local, B_local, C_local, grid);
//     double end = MPI_Wtime();

//     if (rank == 0) {
//         printf("Time for matrix multiplication: %.6f seconds\n", end - start);
//     }

//     // Gather result back into full C of size rowsA x colsB (on rank 0)
//     if (rank == 0) {
//         C_full = malloc((size_t)rowsA * colsB * sizeof(double));
//         if (!C_full) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
//     }
//     gather_2d_blocks_general(C_local, C_full, rowsA, colsB, p_rows, p_cols, grid, 3333);

//     // Write trimmed result (no padding in part (a)) to output file
//     if (rank == 0) {
//         write_matrix_to_file_rect(output_file, C_full, rowsA, colsB);
//         printf("Result written to %s\n", output_file);
//         free(A);
//         free(B);
//         free(C_full);
//     }

//     MPI_Comm_free(&grid);
//     free(A_local);
//     free(B_local);
//     free(C_local);

//     MPI_Finalize();
//     return 0;
// }