/**
 * @file sparse_matrix_reader.h
 * @brief Matrix Market format reader for distributed sparse matrices
 * 
 * @details Functions for reading and distributing sparse matrices and vectors
 * from Matrix Market files across MPI processes.
 */

#ifndef SPARSE_MATRIX_READER_H
#define SPARSE_MATRIX_READER_H

/**
 * @brief Read sparse matrix from Matrix Market format file and distribute it to all processes.
 *
 * Each process reads the portion of matrix rows assigned to itself and stores them locally
 * in Compressed Row Storage (CRS) format. Memory for the CRS format arrays (values, col_indices,
 * row_ptr) will be allocated within this function.
 */
int readSparseMatrix(const char *filename, int num_procs, int rank, int *total_non_zeros, int *local_non_zeros, int *global_rows, int **row_ptr, int **col_indices, double **values);

/**
 * @brief Read right-hand side (RHS) vector b from file and distribute it to all processes.
 */
int readRHSVector(const char *filename, int num_procs, int rank, int global_rows, int local_rows, double *local_b_vector);

/**
 * @brief A debugging function to print the portion of sparse matrix stored on the local process.
 */
int printLocalMatrix(int local_non_zeros, int local_rows, const int *row_ptr, const int *col_indices, const double *values);

/**
 * @brief Find the maximum value in an integer array.
 */
int findMaxInIntArray(const int *array, int size);

#endif // SPARSE_MATRIX_READER_H