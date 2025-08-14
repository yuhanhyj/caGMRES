#ifndef SPARSE_BLAS_H
#define SPARSE_BLAS_H
#include <mpi.h>

int sparseMatrixVectorMultiply(const double *values, const int *col_indices, const int *row_ptr, const double *q_vector, double *u_vector, int local_rows);

int parallelDotProduct(const double *vector_a, const double *vector_b, double *result, int local_size, MPI_Comm *communicator);

int s_step_arnoldi(int s, int k, double **V, double **H, int n_total, const double *M_inv, const double *A, const int *ja, const int *ia, int local_rows, int rank, int num_procs, MPI_Comm *communicator, const int *cntsent, int **to_be_sent, const int *cntowner, int **receive);

#endif // SPARSE_BLAS_H