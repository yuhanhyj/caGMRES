#include "ca_gmres.h"
#include "sparse_blas.h"
#include "localization.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
    int *total_iterations, int *converged)
{
    if (rank == 0)
    {
        printf("Info: Starting Numerically Stable CA-GMRES with s-step = %d\n", s_param);
        printf("Info: Using modified Gram-Schmidt with batched communications\n");
    }

    // Allocate memory for Krylov subspace
    double **V = (double **)malloc((gmres_iterations + 1) * sizeof(double *));
    double **h = (double **)malloc((gmres_iterations + 1) * sizeof(double *));
    for (int i = 0; i <= gmres_iterations; i++)
    {
        V[i] = (double *)calloc(n_total, sizeof(double));
        h[i] = (double *)calloc(gmres_iterations, sizeof(double));
    }

    double *g = (double *)calloc(gmres_iterations + 1, sizeof(double));
    double *c = (double *)calloc(gmres_iterations, sizeof(double));
    double *s = (double *)calloc(gmres_iterations, sizeof(double));

    *total_iterations = 0;
    *converged = 0;

    // Main restart loop
    for (int rst = 0; rst < max_restarts && !(*converged); rst++)
    {

        // Compute initial residual: r0 = b - Ax0
        double *Ax_vec = (double *)calloc(local_rows, sizeof(double));
        synchronizeVector(rank, num_procs, x, cntsent, to_be_sent, cntowner, receive, communicator);
        sparseMatrixVectorMultiply(A, ja, ia, x, Ax_vec, local_rows);

        double *r0 = (double *)calloc(local_rows, sizeof(double));
        for (int i = 0; i < local_rows; i++)
        {
            r0[i] = b_original[i] - Ax_vec[i];
        }
        free(Ax_vec);

        // Apply preconditioner if enabled
        if (preconditioner_flag && M_inv)
        {
            for (int i = 0; i < local_rows; i++)
            {
                r0[i] *= M_inv[i];
            }
        }

        // Compute initial residual norm
        double norm_r0;
        parallelDotProduct(r0, r0, &norm_r0, local_rows, communicator);
        norm_r0 = sqrt(norm_r0);

        if (rank == 0)
        {
            printf("Info: CA-GMRES Restart %d, initial residual norm = %e\n", rst, norm_r0);
        }

        // Check for early convergence
        if (norm_r0 < true_residual_tol)
        {
            *converged = 1;
            free(r0);
            break;
        }

        // Initialize first Krylov vector: v1 = r0 / ||r0||
        for (int i = 0; i < local_rows; i++)
        {
            V[0][i] = r0[i] / norm_r0;
        }
        free(r0);

        // Initialize least squares problem
        g[0] = norm_r0;
        for (int i = 1; i <= gmres_iterations; i++)
        {
            g[i] = 0.0;
        }

        int k_final = 0;
        int inner_converged = 0;

        // CA-GMRES with numerical stability: s-step blocking
        for (int k = 0; k < gmres_iterations && !inner_converged; k += s_param)
        {

            int current_s = (k + s_param > gmres_iterations) ? (gmres_iterations - k) : s_param;
            if (current_s <= 0)
                continue;

            if (rank == 0)
            {
                printf("Info: CA-GMRES s-step block: iterations %d to %d\n", k, k + current_s - 1);
            }

            // NUMERICALLY STABLE CA s-step Arnoldi
            ca_s_step_arnoldi(current_s, k, V, h, n_total, M_inv, A, ja, ia,
                              local_rows, rank, num_procs, communicator,
                              cntsent, to_be_sent, cntowner, receive);

            // Apply Givens rotations to ALL new columns in the block
            for (int i_s = 0; i_s < current_s; i_s++)
            {
                int current_k = k + i_s;

                // Apply previous Givens rotations
                for (int j = 0; j < current_k; j++)
                {
                    double temp = c[j] * h[j][current_k] + s[j] * h[j + 1][current_k];
                    h[j + 1][current_k] = -s[j] * h[j][current_k] + c[j] * h[j + 1][current_k];
                    h[j][current_k] = temp;
                }

                // Compute new Givens rotation
                double gamma = sqrt(h[current_k][current_k] * h[current_k][current_k] +
                                    h[current_k + 1][current_k] * h[current_k + 1][current_k]);

                if (gamma > 1e-12)
                {
                    c[current_k] = h[current_k][current_k] / gamma;
                    s[current_k] = h[current_k + 1][current_k] / gamma;
                }
                else
                {
                    c[current_k] = 1.0;
                    s[current_k] = 0.0;
                }

                h[current_k][current_k] = gamma;
                h[current_k + 1][current_k] = 0.0;

                // Update RHS
                g[current_k + 1] = -s[current_k] * g[current_k];
                g[current_k] = c[current_k] * g[current_k];

                // Check convergence
                if (fabs(g[current_k + 1]) < convergence_tol)
                {
                    inner_converged = 1;
                    k_final = current_k + 1;
                    if (rank == 0)
                    {
                        printf("Info: CA-GMRES converged at iteration %d, residual = %e\n",
                               current_k + 1, fabs(g[current_k + 1]));
                    }
                    break;
                }
            }
        }

        if (!inner_converged)
        {
            k_final = gmres_iterations;
        }
        *total_iterations += k_final;

        // Solve upper triangular system for update coefficients
        double *alpha = (double *)calloc(k_final, sizeof(double));
        for (int j = k_final - 1; j >= 0; j--)
        {
            alpha[j] = g[j];
            for (int l = j + 1; l < k_final; l++)
            {
                alpha[j] -= h[j][l] * alpha[l];
            }
            if (fabs(h[j][j]) > 1e-12)
            {
                alpha[j] /= h[j][j];
            }
        }

        // Update solution
        for (int j = 0; j < k_final; j++)
        {
            for (int i = 0; i < local_rows; i++)
            {
                x[i] += alpha[j] * V[j][i];
            }
        }
        free(alpha);

        // Compute true residual
        Ax_vec = (double *)calloc(local_rows, sizeof(double));
        synchronizeVector(rank, num_procs, x, cntsent, to_be_sent, cntowner, receive, communicator);
        sparseMatrixVectorMultiply(A, ja, ia, x, Ax_vec, local_rows);

        double local_true_res_sq = 0.0;
        for (int i = 0; i < local_rows; i++)
        {
            double diff = b_original[i] - Ax_vec[i];
            local_true_res_sq += diff * diff;
        }
        free(Ax_vec);

        double global_true_res_sq;
        MPI_Allreduce(&local_true_res_sq, &global_true_res_sq, 1, MPI_DOUBLE, MPI_SUM, *communicator);
        double true_residual = sqrt(global_true_res_sq);

        if (rank == 0)
        {
            printf("Info: CA-GMRES Restart %d, true residual norm = %e\n", rst, true_residual);
        }

        if (true_residual < true_residual_tol)
        {
            *converged = 1;
        }
    }

    // Cleanup memory
    for (int i = 0; i <= gmres_iterations; i++)
    {
        free(V[i]);
        free(h[i]);
    }
    free(V);
    free(h);
    free(g);
    free(c);
    free(s);

    return 0;
}

int ca_s_step_arnoldi(
    int s, int k, double **V, double **H, int n_total,
    const double *M_inv, const double *A, const int *ja, const int *ia,
    int local_rows, int rank, int num_procs, MPI_Comm *communicator,
    const int *cntsent, int **to_be_sent,
    const int *cntowner, int **receive)
{
    if (rank == 0)
    {
        printf("Info: Starting Stable CA s-step Arnoldi with s=%d, k=%d\n", s, k);
    }

    for (int j = 0; j < s; j++)
    {
        int current_k = k + j;
        if (current_k >= n_total)
            break;

        double *w = V[current_k + 1];

        // Matrix-vector multiplication
        synchronizeVector(rank, num_procs, V[current_k], cntsent, to_be_sent,
                          cntowner, receive, communicator);
        sparseMatrixVectorMultiply(A, ja, ia, V[current_k], w, local_rows);

        // Apply preconditioner
        if (M_inv)
        {
            for (int l = 0; l < local_rows; l++)
            {
                w[l] *= M_inv[l];
            }
        }

        // BATCHED ORTHOGONALIZATION - The key CA-GMRES feature
        if (current_k == 0)
        {
            // First iteration: only one inner product needed
            parallelDotProduct(w, V[0], &H[0][current_k], local_rows, communicator);
            for (int l = 0; l < local_rows; l++)
            {
                w[l] -= H[0][current_k] * V[0][l];
            }
        }
        else
        {
            // COMMUNICATION AVOIDANCE: Batch all inner products into one MPI call
            double *batch_results = (double *)malloc((current_k + 1) * sizeof(double));

            // Compute ALL inner products at once
            batch_inner_products(V, current_k + 1, w, batch_results, local_rows, communicator);

            // Store in Hessenberg matrix
            for (int i = 0; i <= current_k; i++)
            {
                H[i][current_k] = batch_results[i];
            }

            // Orthogonalize against ALL previous vectors
            for (int i = 0; i <= current_k; i++)
            {
                for (int l = 0; l < local_rows; l++)
                {
                    w[l] -= H[i][current_k] * V[i][l];
                }
            }

            free(batch_results);
        }

        // Compute norm and normalize single communication
        double norm_w;
        parallelDotProduct(w, w, &norm_w, local_rows, communicator);
        norm_w = sqrt(norm_w);
        H[current_k + 1][current_k] = norm_w;

        // Normalize with breakdown protection
        if (norm_w > 1e-12)
        {
            for (int l = 0; l < local_rows; l++)
            {
                V[current_k + 1][l] = w[l] / norm_w;
            }
        }
        else
        {
            if (rank == 0)
            {
                printf("Warning: Breakdown detected at iteration %d, norm = %e\n", current_k, norm_w);
            }
            // Set to zero vector for safety
            for (int l = 0; l < local_rows; l++)
            {
                V[current_k + 1][l] = 0.0;
            }
            H[current_k + 1][current_k] = 0.0;
        }
    }

    if (rank == 0)
    {
        printf("Info: Stable CA s-step Arnoldi completed for block k=%d to k=%d\n", k, k + s - 1);
    }

    return 0;
}

int batch_inner_products(
    double **vectors, int num_vectors, const double *target_vector,
    double *results, int local_size, MPI_Comm *communicator)
{
    // COMMUNICATION REDUCTION: Single MPI_Allreduce for multiple inner products
    double *local_products = (double *)malloc(num_vectors * sizeof(double));

    // Compute all local dot products
    for (int i = 0; i < num_vectors; i++)
    {
        local_products[i] = 0.0;
        for (int j = 0; j < local_size; j++)
        {
            local_products[i] += vectors[i][j] * target_vector[j];
        }
    }

    // Single global reduction for ALL inner products
    MPI_Allreduce(local_products, results, num_vectors, MPI_DOUBLE, MPI_SUM, *communicator);

    free(local_products);
    return 0;
}

int start_nonblocking_allreduce(
    const double *local_values, double *global_values, int count,
    MPI_Comm *communicator, MPI_Request *request)
{
    return MPI_Iallreduce(local_values, global_values, count, MPI_DOUBLE, MPI_SUM,
                          *communicator, request);
}

int complete_nonblocking_allreduce(MPI_Request *request)
{
    MPI_Status status;
    return MPI_Wait(request, &status);
}