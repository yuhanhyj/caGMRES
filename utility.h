#ifndef UTILITY_H
#define UTILITY_H

// Print command line syntax instructions for how to run the program
void printCommandSyntax();

// Print a double precision floating point matrix
// name: Descriptive name of the matrix
// matrixData: Pointer to matrix data (stored in row-major order)
// numRows: Number of rows in the matrix
// numCols: Number of columns in the matrix
void printDoubleMatrix(const char *name, const double *matrixData, int numRows, int numCols);

// Print an integer matrix
void printIntMatrix(const char *name, const int *matrixData, int numRows, int numCols);

// Print a double precision floating point array
// name: Descriptive name of the array
// arrayData: Pointer to array data
// size: Number of elements in the array
void printDoubleArray(const char *name, const double *arrayData, int size);

// Print an integer array
void printIntArray(const char *name, const int *arrayData, int size);

/**
 * @brief Check the orthogonality of a set of distributed vectors, process 0 prints the V^T * V matrix.
 * @param name Descriptive name
 * @param V Array storing the vectors
 * @param num_vectors Number of vectors to check
 * @param local_rows Number of vector rows per process
 * @param n_total Total vector dimension (including ghost points)
 * @param rank Current process number
 * @param communicator MPI communicator
 */
void checkOrthogonality(const char *name, double **V, int num_vectors, int local_rows, int n_total, int rank, MPI_Comm *communicator);
#endif // UTILITY_H