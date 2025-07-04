#ifndef CA_GMRES_H
#define CA_GMRES_H

#include <mpi.h>

/**
 * @brief Communication-Avoiding GMRES solver with s-step method and non-blocking communications
 * 
 * This function implements the CA-GMRES algorithm that reduces global communication
 * by using s-step blocking and delayed orthogonalization techniques.
 * 
 * Key features:
 * - Reduces communication frequency by factor of s
 * - Uses non-blocking MPI operations for overlap
 * - Implements numerically stable orthogonalization
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
    int *total_iterations, int *converged
);

/**
 * @brief S-step Arnoldi process with delayed orthogonalization and communication-computation overlap
 * 
 * This function implements the communication-avoiding s-step Arnoldi method
 * that reduces communication by batching multiple Arnoldi iterations and overlapping
 * communication with computation for improved performance.
 */
int ca_s_step_arnoldi(
    int s, int k, double **V, double **H, int n_total,
    const double *M_inv, const double *A, const int *ja, const int *ia,
    int local_rows, int rank, int num_procs, MPI_Comm *communicator,
    const int *cntsent, int **to_be_sent,
    const int *cntowner, int **receive
);

/**
 * @brief Batch inner product computation with non-blocking communication
 * 
 * Computes multiple inner products simultaneously to reduce communication overhead.
 */
int batch_inner_products(
    double **vectors, int num_vectors, const double *target_vector,
    double *results, int local_size, MPI_Comm *communicator
);

/**
 * @brief Non-blocking communication manager for overlapping computation and communication
 */
int start_nonblocking_allreduce(
    const double *local_values, double *global_values, int count,
    MPI_Comm *communicator, MPI_Request *request
);

/**
 * @brief Wait for non-blocking communication to complete
 */
int complete_nonblocking_allreduce(MPI_Request *request);

/**
 * @brief Start async vector normalization operation
 * 
 * Initiates non-blocking computation of vector 2-norm, allowing computation-communication overlap.
 * The local norm computation is performed immediately, and the global reduction is started
 * asynchronously.
 * 
 * @param vector Input vector for norm computation
 * @param local_norm_sq Output: local contribution to squared norm
 * @param global_norm_sq Output buffer for global squared norm (will be filled when request completes)
 * @param local_size Size of local vector portion
 * @param communicator MPI communicator for global reduction
 * @param request Output: MPI request handle for tracking async operation
 * @return 0 on success, non-zero on error
 */
int start_async_vector_normalization(
    const double *vector, double *local_norm_sq, double *global_norm_sq,
    int local_size, MPI_Comm *communicator, MPI_Request *request
);

/**
 * @brief Complete async vector normalization and finalize vector
 * 
 * Waits for the async norm computation to complete, computes the final norm,
 * and normalizes the vector in-place if the norm is sufficiently large.
 * 
 * @param request MPI request handle from start_async_vector_normalization
 * @param global_norm_sq Input buffer containing global squared norm (from async operation)
 * @param vector Input/Output vector to be normalized in-place
 * @param local_size Size of local vector portion
 * @param final_norm Output: computed final norm value
 * @return 0 on success, non-zero on error
 */
int complete_async_vector_normalization(
    MPI_Request *request, const double *global_norm_sq,
    double *vector, int local_size, double *final_norm
);

/**
 * @brief Check if multiple async requests have completed
 * 
 * Tests multiple MPI requests and returns information about which ones have completed.
 * This is used for managing multiple overlapping operations in the s-step Arnoldi process.
 * 
 * @param requests Array of MPI request handles
 * @param num_requests Number of requests to check
 * @param completed_flags Output: array indicating which requests have completed
 * @param num_completed Output: total number of completed requests
 * @return 0 on success, non-zero on error
 */
int check_multiple_async_requests(
    MPI_Request *requests, int num_requests,
    int *completed_flags, int *num_completed
);

/**
 * @brief Enhanced batch inner products with async capability
 * 
 * Similar to batch_inner_products but with support for non-blocking execution,
 * allowing the caller to overlap this communication with other computations.
 * 
 * @param vectors Array of vectors for inner product computation
 * @param num_vectors Number of vectors in the array
 * @param target_vector Vector to compute inner products against
 * @param local_results Output buffer for local contributions
 * @param global_results Output buffer for global results (filled when request completes)
 * @param local_size Size of local vector portions
 * @param communicator MPI communicator
 * @param request Output: MPI request handle for tracking async operation
 * @return 0 on success, non-zero on error
 */
int batch_inner_products_async(
    double **vectors, int num_vectors, const double *target_vector,
    double *local_results, double *global_results, int local_size,
    MPI_Comm *communicator, MPI_Request *request
);

#endif // CA_GMRES_H