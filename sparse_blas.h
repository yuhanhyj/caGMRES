#ifndef SPARSE_BLAS_H
#define SPARSE_BLAS_H
#include <mpi.h>

// Performance tuning parameters
#define CACHE_LINE_SIZE 64
#define VECTOR_BLOCK_SIZE 512

// Try different BLAS interfaces (CBLAS is preferred but not always available)
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS_H)
#include <cblas.h>
#elif defined(HAVE_BLAS_H) 
extern double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);
extern void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
extern void dscal_(const int *n, const double *alpha, double *x, const int *incx);
extern void dcopy_(const int *n, const double *x, const int *incx, double *y, const int *incy);
#else
// Fallback: use our own implementations
#ifndef USE_CUSTOM_BLAS
#define USE_CUSTOM_BLAS
#endif
#endif

int sparseMatrixVectorMultiply(const double *values, const int *col_indices, const int *row_ptr, const double *q_vector, double *u_vector, int local_rows);

int parallelDotProduct(const double *vector_a, const double *vector_b, double *result, int local_size, MPI_Comm *communicator);

int s_step_arnoldi(int s, int k, double **V, double **H, int n_total, const double *M_inv, const double *A, const int *ja, const int *ia, int local_rows, int rank, int num_procs, MPI_Comm *communicator, const int *cntsent, int **to_be_sent, const int *cntowner, int **receive);

// New optimized BLAS functions
int batchDotProducts(double **vectors_a, double **vectors_b, double *results, int num_pairs, int local_size, MPI_Comm *communicator);
int optimizedBatchSpMV(const double *values, const int *col_indices, const int *row_ptr, double **input_vectors, double **output_vectors, int num_vectors, int local_rows);
int blasDotProduct(const double *vector_a, const double *vector_b, double *result, int local_size);
int blasVectorUpdate(double *w, const double *v, double alpha, int local_size);
int blasVectorScale(double *vector, double scale, int local_size);

#endif // SPARSE_BLAS_H