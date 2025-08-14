#include "sparse_matrix_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// to ensure maximum compatibility and fix the fscanf segmentation fault.

int readSparseMatrix(const char *filename, int num_procs, int rank, int *total_non_zeros, int *local_non_zeros, int *global_rows, int **row_ptr, int **col_indices, double **values)
{

    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        if (rank == 0)
            fprintf(stderr, "Error: Could not open matrix file '%s'.\n", filename);
        return 1;
    }

    // Use the original's simple fscanf for the header, assuming no comment lines.
    int matrix_cols_ignored; // The original code assumes a square matrix
    fscanf(file, "%d %d %d\n", global_rows, &matrix_cols_ignored, total_non_zeros);

    const int base_rows = *global_rows / num_procs;
    const int extra_rows = *global_rows % num_procs;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int start_row = rank * base_rows + (rank < extra_rows ? rank : extra_rows);
    const int end_row = start_row + local_rows;

    int *non_zeros_per_row = (int *)calloc(local_rows, sizeof(int));
    if (!non_zeros_per_row)
    {
        return 1;
    }

    // First Pass: Count non-zeros per local row
    int i_row, i_col;
    double val;
    for (int i = 0; i < *total_non_zeros; i++)
    {
        fscanf(file, "%d %d %lg\n", &i_row, &i_col, &val);
        i_row--; // Convert to 0-based index
        if (i_row >= start_row && i_row < end_row)
        {
            non_zeros_per_row[i_row - start_row]++;
        }
    }

    // Allocate memory for CRS format
    *local_non_zeros = 0;
    *row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));
    if (!(*row_ptr))
    {
        free(non_zeros_per_row);
        fclose(file);
        return 1;
    }

    for (int i = 0; i < local_rows; i++)
    {
        (*row_ptr)[i] = *local_non_zeros;
        *local_non_zeros += non_zeros_per_row[i];
    }
    (*row_ptr)[local_rows] = *local_non_zeros;

    *values = (double *)malloc(*local_non_zeros * sizeof(double));
    *col_indices = (int *)malloc(*local_non_zeros * sizeof(int));
    if (!(*values) || !(*col_indices))
    { /* Cleanup and exit */
        return 1;
    }

    // Second Pass: Fill CRS arrays
    rewind(file);
    fscanf(file, "%d %d %d\n", &matrix_cols_ignored, &matrix_cols_ignored, &matrix_cols_ignored);

    // Reset non_zeros_per_row to use as an offset counter
    for (int i = 0; i < local_rows; i++)
        non_zeros_per_row[i] = 0;

    for (int i = 0; i < *total_non_zeros; i++)
    {
        fscanf(file, "%d %d %lg\n", &i_row, &i_col, &val);
        i_row--;
        i_col--; // Convert to 0-based
        if (i_row >= start_row && i_row < end_row)
        {
            int local_row = i_row - start_row;
            int insert_pos = (*row_ptr)[local_row] + non_zeros_per_row[local_row];
            (*values)[insert_pos] = val;
            (*col_indices)[insert_pos] = i_col;
            non_zeros_per_row[local_row]++;
        }
    }

    fclose(file);
    free(non_zeros_per_row);
    return 0;
}

// The other functions in the file remain the same...

int findMaxInIntArray(const int *array, int size)
{
    if (array == NULL || size <= 0)
        return -1;
    int max_val = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max_val)
            max_val = array[i];
    }
    return max_val;
}

/**
 * @brief Load the RHS vector b from the file and scatter it across all processes."​​(Uses "scatter," a common MPI term for distributing data.)

 */
int readRHSVector(const char *filename, int num_procs, int rank, int global_rows, int local_rows, double *local_b_vector)
{
    double *global_b_vector = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        FILE *file = fopen(filename, "r");
        if (file == NULL)
        {
            fprintf(stderr, "Error: Could not open RHS file '%s'.\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        global_b_vector = (double *)malloc(global_rows * sizeof(double));
        if (global_b_vector == NULL)
        {
            fprintf(stderr, "Error: Memory allocation failed for global RHS vector.\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        for (int i = 0; i < global_rows; i++)
        {
            if (fscanf(file, "%lg", &global_b_vector[i]) != 1)
            {
                fprintf(stderr, "Error: Failed to read value from RHS file at line %d.\n", i + 1);
                free(global_b_vector);
                fclose(file);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        fclose(file);

        sendcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));
        int base_rows = global_rows / num_procs;
        int extra_rows = global_rows % num_procs;
        int offset = 0;
        for (int i = 0; i < num_procs; i++)
        {
            sendcounts[i] = base_rows + (i < extra_rows ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(
        global_b_vector,
        sendcounts,
        displs,
        MPI_DOUBLE,
        local_b_vector,
        local_rows,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(global_b_vector);
        free(sendcounts);
        free(displs);
    }

    return 0;
}

int printLocalMatrix(int local_non_zeros, int local_rows, const int *row_ptr, const int *col_indices, const double *values)
{

    return 0;
}