#include "sparse_blas.h"
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "localization.h"
#include "communication_profiler.h"

// Original functions remain unchanged
int sparseMatrixVectorMultiply(const double *values, const int *col_indices, const int *row_ptr, const double *q_vector, double *u_vector, int local_rows)
{
    for (int i = 0; i < local_rows; i++)
    {
        u_vector[i] = 0.0;
        const int start_idx = row_ptr[i];
        const int end_idx = row_ptr[i + 1];
        for (int j = start_idx; j < end_idx; j++)
        {
            u_vector[i] += values[j] * q_vector[col_indices[j]];
        }
    }
    return 0;
}

int parallelDotProduct1(const double *vector_a, const double *vector_b, double *result, int local_size, MPI_Comm *communicator)
{
    double local_dot_product = 0.0;
    for (int i = 0; i < local_size; i++)
    {
        local_dot_product += vector_a[i] * vector_b[i];
    }
    MPI_Allreduce(&local_dot_product, result, 1, MPI_DOUBLE, MPI_SUM, *communicator);
    return 0;
}

int parallelDotProduct(const double *vector_a, const double *vector_b, double *result, int local_size, MPI_Comm *communicator)
{

    int rank;
    MPI_Comm_rank(*communicator, &rank);

    // Computation start time
    double comp_start = MPI_Wtime();

    // Local computation
    double local_dot_product = 0.0;
    for (int i = 0; i < local_size; i++)
    {
        local_dot_product += vector_a[i] * vector_b[i];
    }

    // Record computation time
    double comp_time = MPI_Wtime() - comp_start;
    recordComputation(comp_time);

    // Communication start time
    double comm_start = MPI_Wtime();
    MPI_Allreduce(&local_dot_product, result, 1, MPI_DOUBLE, MPI_SUM, *communicator);
    double comm_end = MPI_Wtime();

    if (rank == 0)
    {
        recordInnerProductCommunication(comm_end - comm_start);
    }

    return 0;
}

// Original s_step_arnoldi for classical GMRES (with profiling)
int s_step_arnoldi(int s, int k, double **V, double **H, int n_total, const double *M_inv, const double *A, const int *ja, const int *ia, int local_rows, int rank, int num_procs, MPI_Comm *communicator, const int *cntsent, int **to_be_sent, const int *cntowner, int **receive)
{
    for (int i = 0; i < s; ++i)
    {
        int current_k = k + i;
        double *w = V[current_k + 1];

        // Record iteration
        recordIteration();

        synchronizeVector(rank, num_procs, V[current_k], cntsent, to_be_sent, cntowner, receive, communicator);

        // Computation start time
        double comp_start = MPI_Wtime();
        sparseMatrixVectorMultiply(A, ja, ia, V[current_k], w, local_rows);
        if (M_inv)
        {
            for (int j = 0; j < local_rows; j++)
                w[j] *= M_inv[j];
        }
        // Record computation time
        double comp_time = MPI_Wtime() - comp_start;
        recordComputation(comp_time);

        for (int j = 0; j <= current_k; j++)
        {
            parallelDotProduct(w, V[j], &H[j][current_k], local_rows, communicator);

            // Vector update computation time
            double update_start = MPI_Wtime();
            for (int l = 0; l < local_rows; l++)
                w[l] -= H[j][current_k] * V[j][l];
            double update_time = MPI_Wtime() - update_start;
            recordComputation(update_time);
        }

        parallelDotProduct(w, w, &H[current_k + 1][current_k], local_rows, communicator);
        H[current_k + 1][current_k] = sqrt(H[current_k + 1][current_k]);

        if (H[current_k + 1][current_k] > 1e-12)
        {
            // Normalization computation time
            double norm_start = MPI_Wtime();
            for (int l = 0; l < local_rows; l++)
                w[l] /= H[current_k + 1][current_k];
            double norm_time = MPI_Wtime() - norm_start;
            recordComputation(norm_time);
        }
    }
    return 0;
}

// Additional utility functions for CA-GMRES support

/**
 * @brief Batch matrix-vector multiplication for multiple vectors
 *
 * This function can be used by CA-GMRES to efficiently compute multiple
 * matrix-vector products, which is useful in the s-step Arnoldi process.
 */
int batchMatrixVectorMultiply(
    const double *values, const int *col_indices, const int *row_ptr,
    double **input_vectors, double **output_vectors, int num_vectors, int local_rows)
{
    for (int vec = 0; vec < num_vectors; vec++)
    {
        sparseMatrixVectorMultiply(values, col_indices, row_ptr,
                                   input_vectors[vec], output_vectors[vec], local_rows);
    }
    return 0;
}

/**
 * @brief Modified Gram-Schmidt orthogonalization for a single vector against multiple vectors
 *
 * This function performs orthogonalization of vector w against all vectors in V[0:k],
 * which is commonly used in CA-GMRES delayed orthogonalization.

 */
int modifiedGramSchmidt(
    double *w, double **V, double *h, int k, int local_size, MPI_Comm *communicator)
{
    for (int j = 0; j < k; j++)
    {
        // Compute inner product h[j] = <w, V[j]>
        parallelDotProduct(w, V[j], &h[j], local_size, communicator);

        // Orthogonalize: w = w - h[j] * V[j]
        for (int i = 0; i < local_size; i++)
        {
            w[i] -= h[j] * V[j][i];
        }
    }
    return 0;
}

/**
 * @brief Compute vector 2-norm with parallel reduction
 *
 */
int parallelVectorNorm(const double *vector, double *norm, int local_size, MPI_Comm *communicator)
{
    double local_norm_sq = 0.0;
    for (int i = 0; i < local_size; i++)
    {
        local_norm_sq += vector[i] * vector[i];
    }

    double global_norm_sq;
    MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, *communicator);
    *norm = sqrt(global_norm_sq);

    return 0;
}

/**
 * @brief Vector scaling: scale vector by a constant
 *
 */
int scaleVector(double *vector, double scale, int local_size)
{
    for (int i = 0; i < local_size; i++)
    {
        vector[i] *= scale;
    }
    return 0;
}

/**
 * @brief Vector copy: dst = src
 */
int copyVector(double *dst, const double *src, int local_size)
{
    for (int i = 0; i < local_size; i++)
    {
        dst[i] = src[i];
    }
    return 0;
}

/**
 * @brief Vector addition: dst = dst + alpha * src
 */
int addScaledVector(double *dst, const double *src, double alpha, int local_size)
{
    for (int i = 0; i < local_size; i++)
    {
        dst[i] += alpha * src[i];
    }
    return 0;
}