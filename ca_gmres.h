#ifndef CA_GMRES_H
#define CA_GMRES_H

#include <mpi.h>

/**
 * @brief Communication-Avoiding GMRES solver with s-step method and non-blocking communications
 *
 * This function implements the CA-GMRES algorithm that reduces global communication
 * by using s-step blocking and delayed orthogonalization techniques.
 */
int ca_gmres_solve(
    const double *A, const int *ja, const int *ia,
    const double *b_original, double *x,
    int local_rows, int n_total, int global_rows,
    int s_param, int gmres_iterations, int max_restarts,
    double convergence_tol, double true_residual_tol,
    int preconditioner_flag, const double *M_inv,
    int rank, int num_procs, MPI_Comm *communicator,
    const int *cntsent, int **to_be_sent,
    const int *cntowner, int **receive,
    int *total_iterations, int *converged);

/**
 * @brief S-step Arnoldi process with delayed orthogonalization
 *
 * This function implements the communication-avoiding s-step Arnoldi method
 * that reduces communication by batching multiple Arnoldi iterations.
 */
int ca_s_step_arnoldi(
    int s, int k, double **V, double **H, int n_total,
    const double *M_inv, const double *A, const int *ja, const int *ia,
    int local_rows, int rank, int num_procs, MPI_Comm *communicator,
    const int *cntsent, int **to_be_sent,
    const int *cntowner, int **receive);

/**
 * @brief Batch inner product computation with non-blocking communication
 *
 * Computes multiple inner products simultaneously to reduce communication overhead.

 */
int batch_inner_products(
    double **vectors, int num_vectors, const double *target_vector,
    double *results, int local_size, MPI_Comm *communicator);

/**
 * @brief Non-blocking communication manager for overlapping computation and communication

 */
int start_nonblocking_allreduce(
    const double *local_values, double *global_values, int count,
    MPI_Comm *communicator, MPI_Request *request);

/**
 * @brief Wait for non-blocking communication to complete
 */
int complete_nonblocking_allreduce(MPI_Request *request);

#endif // CA_GMRES_H