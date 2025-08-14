#include <stdio.h>       // For printf function
#include <stdlib.h>      // C standard library, usually included in large projects for future needs
#include <mpi.h>         // Ensure mpi.h is included
#include "sparse_blas.h" // Need parallelDotProduct
// Include our own header file to ensure implementation matches declaration
#include "utility.h"

/**
 * @brief Print command line usage instructions for the program.
 * This function is called when the user provides incorrect or insufficient arguments.
 */
void printCommandSyntax()
{
    // Using fprintf(stderr, ...) to output error messages to standard error stream is a good practice.
    // \n is a newline character, making the output cleaner.
    // Note: The usage instructions here have been updated according to the actual situation in README.md and mpi_gmres.c.
    fprintf(stderr, "\nError: Not enough or incorrect input arguments.\n");
    fprintf(stderr, "Syntax: mpirun -np <processes> ./gmres <matrix_file> <rhs_file> <iterations> <preconditioner_flag> <restarts>\n\n");
}

/**
 * @brief Print a double precision floating point matrix.
 * @param name Descriptive name of the matrix, used for identification in output.
 * @param matrixData Pointer to matrix data (stored in row-major order).
 * @param numRows Number of rows in the matrix.
 * @param numCols Number of columns in the matrix.
 */
void printDoubleMatrix(const char *name, const double *matrixData, int numRows, int numCols)
{
    printf("\n--- Matrix: %s ---\n", name);
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            // Calculate index in one-dimensional array using row-major order
            int index = i * numCols + j;

            // Use %14.7f for formatted output, total width 14, 7 decimal places, easier to align and read
            printf("%14.7f", matrixData[index]);

            // Print a comma and space as long as it's not the last element in the row
            if (j < numCols - 1)
            {
                printf(", ");
            }
        }
        printf("\n"); // Line break after printing each row
    }
    printf("--- End of Matrix: %s ---\n\n", name);
}

/**
 * @brief Print an integer matrix.
 * (Function structure is very similar to printDoubleMatrix)
 */
void printIntMatrix(const char *name, const int *matrixData, int numRows, int numCols)
{
    printf("\n--- Matrix: %s ---\n", name);
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            int index = i * numCols + j;
            // %8d means print an integer with width 8 for alignment
            printf("%8d", matrixData[index]);
            if (j < numCols - 1)
            {
                printf(", ");
            }
        }
        printf("\n");
    }
    printf("--- End of Matrix: %s ---\n\n", name);
}

/**
 * @brief Print a double precision floating point one-dimensional array.
 * @param name Descriptive name of the array.
 * @param arrayData Pointer to array data.
 * @param size Number of elements in the array.
 */
void printDoubleArray(const char *name, const double *arrayData, int size)
{
    printf("\n--- Array: %s ---\n", name);
    for (int i = 0; i < size; i++)
    {
        // Print index and value of each element
        printf("%s[%d] = %f\n", name, i, arrayData[i]);
    }
    printf("--- End of Array: %s ---\n\n", name);
}

/**
 * @brief Print an integer one-dimensional array.
 * (Function structure is very similar to printDoubleArray)
 */
void printIntArray(const char *name, const int *arrayData, int size)
{
    printf("\n--- Array: %s ---\n", name);
    for (int i = 0; i < size; i++)
    {
        printf("%s[%d] = %d\n", name, i, arrayData[i]);
    }
    printf("--- End of Array: %s ---\n\n", name);
}

void checkOrthogonality(const char *name, double **V, int num_vectors, int local_rows, int n_total, int rank, MPI_Comm *communicator)
{
    if (num_vectors <= 0)
        return;

    // Only process 0 is responsible for allocation, calculation and printing
    double *vtv_matrix = NULL;
    if (rank == 0)
    {
        vtv_matrix = (double *)malloc(num_vectors * num_vectors * sizeof(double));
        if (!vtv_matrix)
        {
            MPI_Abort(*communicator, 1);
        }
    }

    // All processes collaboratively compute V[i]·V[j]
    for (int i = 0; i < num_vectors; i++)
    {
        for (int j = 0; j < num_vectors; j++)
        {
            double dot_product;
            parallelDotProduct(V[i], V[j], &dot_product, local_rows, communicator);

            if (rank == 0)
            {
                vtv_matrix[i * num_vectors + j] = dot_product;
            }
        }
    }

    // Process 0 prints the results
    if (rank == 0)
    {
        printDoubleMatrix(name, vtv_matrix, num_vectors, num_vectors);
        free(vtv_matrix);
    }
}