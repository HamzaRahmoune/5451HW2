#define _POSIX_C_SOURCE 200809L  // enables getline() and ssize_t on most compilers
#include <sys/types.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* ---------- Matrix reading from file (use streaming for very large files >500MB) ---------- */
void read_matrix_streaming(const char *filename, double **A, double **B, int *n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int readingA = 0, readingB = 0;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

    // First pass — determine dimensions
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

    rewind(fp);

    *n = rowsA;
    *A = malloc(rowsA * colsA * sizeof(double));
    *B = malloc(rowsB * colsB * sizeof(double));

    int idxA = 0, idxB = 0;
    readingA = readingB = 0;

    // Second pass — parse data directly into flat arrays
    while ((read = getline(&line, &len, fp)) != -1) {
        if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
        if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
        if (!readingA && !readingB) continue;

        char *tmp = strtok(line, " \t\n");
        while (tmp) {
            double val = atof(tmp);
            if (readingA) (*A)[idxA++] = val;
            else if (readingB) (*B)[idxB++] = val;
            tmp = strtok(NULL, " \t\n");
        }
    }

    free(line);
    fclose(fp);
}


/* ---------- Local matrix multiplication ---------- */
void MatrixMultiply(int nlocal, double *A, double *B, double *C) {
    for (int i = 0; i < nlocal; i++)
        for (int j = 0; j < nlocal; j++)
            for (int k = 0; k < nlocal; k++)
                C[i*nlocal + j] += A[i*nlocal + k] * B[k*nlocal + j];
}

/* ---------- Write matrix to output file ---------- */
void write_matrix_to_file(const char *filename, double *C, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%.4f", C[i * n + j]);
            if (j < n - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* ---------- Debug: Print matrix ---------- */
void PrintMatrix(int n, int m, double* mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%8.2f ", mat[i * m + j]);
        }
        printf("\n");
    }
}

/* ---------- Create Cartesian grid communicator ---------- */
MPI_Comm create_cartesian_grid(MPI_Comm base_comm, int *nlocal_out, int n)
{
    int npes;
    MPI_Comm_size(base_comm, &npes);

    int dims[2] = {0, 0};
    MPI_Dims_create(npes, 2, dims);

    int periods[2] = {1, 1};  /* wraparound grid for Cannon */
    MPI_Comm grid;
    MPI_Cart_create(base_comm, 2, dims, periods, 1, &grid);

    int nlocal = n / dims[0];
    if (nlocal_out)
        *nlocal_out = nlocal;

    return grid;
}

/* ---------- Scatter global n×n matrix into nlocal×nlocal blocks ---------- */
void scatter_2d_blocks(double *A_global, double *local,
                       int n, MPI_Comm grid)
{
    int npes, rank;
    MPI_Comm_size(grid, &npes);
    MPI_Comm_rank(grid, &rank);

    int dims[2], coords[2];
    MPI_Cart_get(grid, 2, dims, (int[]){1,1}, coords);
    int nlocal = n / dims[0];

    if (rank == 0) {
        for (int r = 0; r < npes; r++) {
            int rc[2];
            MPI_Cart_coords(grid, r, 2, rc);

            int sizes[2]    = { n,      n      };
            int subsizes[2] = { nlocal, nlocal };
            int starts[2]   = { rc[0]*nlocal, rc[1]*nlocal };

            MPI_Datatype sub;
            MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                     MPI_ORDER_C, MPI_DOUBLE, &sub);
            MPI_Type_commit(&sub);

            if (r == 0) {
                for (int i = 0; i < nlocal; i++) {
                    const double *src = A_global + (starts[0]+i)*n + starts[1];
                    double *dst = local + i*nlocal;
                    memcpy(dst, src, nlocal * sizeof(double));
                }
            } else {
                MPI_Send(A_global, 1, sub, r, 999, grid);
            }
            MPI_Type_free(&sub);
        }
    } else {
        MPI_Recv(local, nlocal*nlocal, MPI_DOUBLE, 0, 999, grid, MPI_STATUS_IGNORE);
    }
}

/* ---------- Gather nlocal×nlocal blocks back into global n×n matrix ---------- */
void gather_2d_blocks(double *local, double *C_global,
                      int n, MPI_Comm grid)
{
    int npes, rank;
    MPI_Comm_size(grid, &npes);
    MPI_Comm_rank(grid, &rank);

    int dims[2], coords[2];
    MPI_Cart_get(grid, 2, dims, (int[]){1,1}, coords);
    int nlocal = n / dims[0];

    if (rank == 0) {
        for (int r = 0; r < npes; r++) {
            int rc[2];
            MPI_Cart_coords(grid, r, 2, rc);

            int sizes[2]    = { n,      n      };
            int subsizes[2] = { nlocal, nlocal };
            int starts[2]   = { rc[0]*nlocal, rc[1]*nlocal };

            MPI_Datatype sub;
            MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                     MPI_ORDER_C, MPI_DOUBLE, &sub);
            MPI_Type_commit(&sub);

            if (r == 0) {
                for (int i = 0; i < nlocal; i++) {
                    double *dst = C_global + (starts[0]+i)*n + starts[1];
                    const double *src = local + i*nlocal;
                    memcpy(dst, src, nlocal * sizeof(double));
                }
            } else {
                MPI_Recv(C_global, 1, sub, r, 1999, grid, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&sub);
        }
    } else {
        MPI_Send(local, nlocal*nlocal, MPI_DOUBLE, 0, 1999, grid);
    }
}

void MatrixMatrixMultiply(int n, double *a, double *b, double *c, MPI_Comm grid)
{
    int i;
    int nlocal;
    int npes, myrank, mycoords[2];
    int dims[2], periods[2];
    int uprank, downrank, leftrank, rightrank;
    int shiftsource, shiftdest;
    MPI_Status status;

    /* Query communicator info from the existing grid */
    MPI_Comm_size(grid, &npes);
    MPI_Comm_rank(grid, &myrank);
    MPI_Cart_get(grid, 2, dims, periods, mycoords);

    nlocal = n / dims[0];

    /* Determine neighbors for shifts */
    MPI_Cart_shift(grid, 1, +1, &leftrank, &rightrank);  // left/right along columns
    MPI_Cart_shift(grid, 0, +1, &uprank, &downrank);     // up/down along rows

    /* ---------- Initial alignment ---------- */
    MPI_Cart_shift(grid, 1, -mycoords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_DOUBLE,
                         shiftdest, 100,
                         shiftsource, 100, grid, &status);

    MPI_Cart_shift(grid, 0, -mycoords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_DOUBLE,
                         shiftdest, 200,
                         shiftsource, 200, grid, &status);


    /* ---------- Main Cannon loop ---------- */
    for (i = 0; i < dims[0]; i++) {
        MatrixMultiply(nlocal, a, b, c);

        /* Rotate A left */
        MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_DOUBLE,
                             leftrank, 1000 + i,
                             rightrank, 1000 + i, grid, &status);

        /* Rotate B up */
        MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_DOUBLE,
                             uprank, 2000 + i,
                             downrank, 2000 + i, grid, &status);
    }

    /* ---------- Restore original distribution ---------- */
    MPI_Cart_shift(grid, 1, +mycoords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_DOUBLE,
                         shiftdest, 300,
                         shiftsource, 300, grid, &status);

    MPI_Cart_shift(grid, 0, +mycoords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_DOUBLE,
                         shiftdest, 400,
                         shiftsource, 400, grid, &status);
}



/* ---------- Main ---------- */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    double *A = NULL, *B = NULL, *C = NULL, *C_full = NULL;
    int n, nlocal;

    if (rank == 0)
        read_matrix_streaming(input_file, &A, &B, &n);

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Create Cartesian grid once and reuse */
    MPI_Comm grid = create_cartesian_grid(MPI_COMM_WORLD, &nlocal, n);
    
    double *a_local = malloc(nlocal * nlocal * sizeof(double));
    double *b_local = malloc(nlocal * nlocal * sizeof(double));
    C = calloc(nlocal * nlocal, sizeof(double));

    /* Proper 2D scatter */
    scatter_2d_blocks(A, a_local, n, grid);
    scatter_2d_blocks(B, b_local, n, grid);

    double start, end, elapsed;
    start = MPI_Wtime();
    /* Run Cannon’s algorithm */
    MatrixMatrixMultiply(n, a_local, b_local, C, grid);
    end = MPI_Wtime();
    if (rank == 0) {
        elapsed = end - start;
        printf("Time for matrix multiplication: %.6f seconds\n", elapsed);
    }

    /* Gather result back */
    if (rank == 0)
        C_full = malloc(n * n * sizeof(double));
    gather_2d_blocks(C, C_full, n, grid);

    /* Write result to output file */
    if (rank == 0) {
        write_matrix_to_file(output_file, C_full, n);
        printf("Result written to %s\n", output_file);
        free(A);
        free(B);
        free(C_full);
    }

    MPI_Comm_free(&grid);
    free(a_local);
    free(b_local);
    free(C);

    MPI_Finalize();
    return 0;
}