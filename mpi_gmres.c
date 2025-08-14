#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "communication_profiler.h"

#include "utility.h"
#include "sparse_matrix_reader.h"
#include "localization.h"
#include "sparse_blas.h"
#include "ca_gmres.h"

void cleanup(int num_procs, int gmres_iters,
             double *A, int *ia, int *ja, double *b, double *x,
             double **V, double **h, double *g, double *c, double *s,
             int *cntsent, int **to_be_sent, int *cntowner, int **receive, double *M_inv,
             double *b_original, double *X)
{
    free(A);
    free(ia);
    free(ja);
    free(b);
    free(x);
    free(X);
    free(g);
    free(c);
    free(s);
    if (M_inv)
        free(M_inv);
    if (b_original)
        free(b_original);
    for (int i = 0; i <= gmres_iters; i++)
    {
        if (V[i])
            free(V[i]);
        if (h[i])
            free(h[i]);
    }
    free(V);
    free(h);
    free(cntsent);
    free(cntowner);
    if (to_be_sent)
    {
        for (int i = 0; i < num_procs; ++i)
            if (to_be_sent[i])
                free(to_be_sent[i]);
        free(to_be_sent);
    }
    if (receive)
    {
        for (int i = 0; i < num_procs; ++i)
            if (receive[i])
                free(receive[i]);
        free(receive);
    }
}

int main(int argc, char *argv[])
{
    int rank = 0, num_procs = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm communicator = MPI_COMM_WORLD;

    // Initialize communication performance profiler
    initCommunicationProfiler(rank, num_procs);

    if (argc < 6)
    {
        if (rank == 0)
            printCommandSyntax();
        MPI_Abort(communicator, 1);
    }

    int i, j, ss;
    const char *rhs_filename = argv[2];
    const int gmres_iterations = atoi(argv[3]);
    const int preconditioner_flag = atoi(argv[4]);
    const int max_restarts = atoi(argv[5]);
    const double convergence_tol = (argc >= 8) ? atof(argv[7]) : 1.e-10;
    const double true_residual_tol = 1.e-9;
    const int s_param = (argc >= 7) ? atoi(argv[6]) : 1;

    // Determine which algorithm to use based on s_param
    int use_ca_gmres = (s_param > 1) ? 1 : 0;

    if (rank == 0)
    {
        if (use_ca_gmres)
        {
            printf("Info: Using CA-GMRES algorithm with s-step = %d\n", s_param);
            printf("Info: Features: delayed orthogonalization, non-blocking communications\n");
        }
        else
        {
            printf("Info: Using Classical GMRES algorithm (s-step = %d)\n", s_param);
        }
        printf("Info: max_restarts = %d, preconditioner = %s\n",
               max_restarts, preconditioner_flag ? "ON" : "OFF");
    }

    double t_start, t_end;

    int nnz, local_non_zeros, global_rows;
    int *ia = NULL, *ja = NULL;
    double *A = NULL;

    if (readSparseMatrix(argv[1], num_procs, rank, &nnz, &local_non_zeros, &global_rows, &ia, &ja, &A) != 0)
    {
        MPI_Barrier(communicator);
        if (rank == 0)
            fprintf(stderr, "Matrix reading failed. Aborting all processes.\n");
        MPI_Abort(communicator, 1);
    }
    const int base_rows = global_rows / num_procs;
    const int extra_rows = global_rows % num_procs;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);

    double *b_original = (double *)calloc(local_rows, sizeof(double));
    readRHSVector(rhs_filename, num_procs, rank, global_rows, local_rows, b_original);

    int **receive = NULL, *cntowner = NULL;
    int **to_be_sent = NULL, *cntsent = NULL;
    setupCommunication(rank, num_procs, global_rows, ja, ia, local_non_zeros,
                       &communicator, &cntsent, &to_be_sent, &cntowner, &receive);

    int n_total = (ja && local_non_zeros > 0) ? findMaxInIntArray(ja, local_non_zeros) + 1 : local_rows;

    double *M_inv = NULL;
    if (preconditioner_flag)
    {
        M_inv = (double *)malloc(local_rows * sizeof(double));
        for (i = 0; i < local_rows; i++)
        {
            double diag = 0.0;
            for (j = ia[i]; j < ia[i + 1]; j++)
            {
                if (ja[j] == i)
                {
                    diag = A[j];
                    break;
                }
            }
            M_inv[i] = (fabs(diag) > 1e-16) ? 1.0 / diag : 1.0;
        }
    }

    double *x = (double *)calloc(n_total, sizeof(double));
    double *X = (double *)calloc(n_total, sizeof(double)); // Update vector

    int completed = 0;
    int total_iterations = 0;

    t_start = MPI_Wtime();

    // Choose algorithm based on s_param
    if (use_ca_gmres)
    {
        // Use CA-GMRES algorithm
        if (rank == 0)
        {
            printf("\n=== Starting CA-GMRES Solver ===\n");
        }

        int ca_converged = 0;
        int ca_total_iterations = 0;

        int result = ca_gmres_solve(
            A, ja, ia, b_original, x,
            local_rows, n_total, global_rows,
            s_param, gmres_iterations, max_restarts,
            convergence_tol, true_residual_tol,
            preconditioner_flag, M_inv,
            rank, num_procs, &communicator,
            cntsent, to_be_sent, cntowner, receive,
            &ca_total_iterations, &ca_converged);

        if (result == 0)
        {
            completed = ca_converged;
            total_iterations = ca_total_iterations;
        }
        else
        {
            if (rank == 0)
            {
                fprintf(stderr, "Error: CA-GMRES solver failed\n");
            }
            MPI_Abort(communicator, 1);
        }

        // Leave V, h, g, c, s as NULL for CA-GMRES since it manages its own memory
    }
    else
    {
        // Use Classical GMRES algorithm
        if (rank == 0)
        {
            printf("\n=== Starting Classical GMRES Solver ===\n");
        }

        // Allocate memory for classical GMRES (local scope)
        double **V = (double **)malloc((gmres_iterations + 1) * sizeof(double *));
        double **h = (double **)malloc((gmres_iterations + 1) * sizeof(double *));
        for (i = 0; i <= gmres_iterations; i++)
        {
            V[i] = (double *)calloc(n_total, sizeof(double));
            h[i] = (double *)calloc(gmres_iterations, sizeof(double));
        }

        double *g = (double *)calloc(gmres_iterations + 1, sizeof(double));
        double *c = (double *)calloc(gmres_iterations, sizeof(double));
        double *s = (double *)calloc(gmres_iterations, sizeof(double));

        // Classical GMRES restart loop
        for (int rst = 0; rst < max_restarts && !completed; rst++)
        {
            double *Ax_vec = (double *)calloc(local_rows, sizeof(double));
            synchronizeVector(rank, num_procs, x, cntsent, to_be_sent, cntowner, receive, &communicator);
            sparseMatrixVectorMultiply(A, ja, ia, x, Ax_vec, local_rows);
            double *r0 = (double *)calloc(local_rows, sizeof(double));
            for (i = 0; i < local_rows; i++)
                r0[i] = b_original[i] - Ax_vec[i];
            free(Ax_vec);

            if (preconditioner_flag)
            {
                for (i = 0; i < local_rows; i++)
                    r0[i] *= M_inv[i];
            }

            double norm_r0;
            parallelDotProduct(r0, r0, &norm_r0, local_rows, &communicator);
            norm_r0 = sqrt(norm_r0);

            if (rank == 0)
            {
                printf("Info: Classical GMRES Restart %d, initial preconditioned residual norm = %e\n", rst, norm_r0);
            }

            for (i = 0; i < local_rows; i++)
                V[0][i] = r0[i] / norm_r0;
            free(r0);

            g[0] = norm_r0;
            for (i = 1; i <= gmres_iterations; i++)
                g[i] = 0.0;

            int k_inner_final = 0;
            int inner_completed = 0;

            for (int k = 0; k < gmres_iterations && !inner_completed; k += s_param)
            {
                int current_s = (k + s_param > gmres_iterations) ? (gmres_iterations - k) : s_param;
                if (current_s <= 0)
                    continue;

                s_step_arnoldi(current_s, k, V, h, n_total, M_inv, A, ja, ia, local_rows, rank, num_procs, &communicator, cntsent, to_be_sent, cntowner, receive);

                for (int i_s = 0; i_s < current_s; ++i_s)
                {
                    int current_k = k + i_s;
                    for (j = 0; j < current_k; j++)
                    {
                        double temp = c[j] * h[j][current_k] + s[j] * h[j + 1][current_k];
                        h[j + 1][current_k] = -s[j] * h[j][current_k] + c[j] * h[j + 1][current_k];
                        h[j][current_k] = temp;
                    }

                    double gamma = sqrt(h[current_k][current_k] * h[current_k][current_k] + h[current_k + 1][current_k] * h[current_k + 1][current_k]);
                    c[current_k] = (gamma > 1e-12) ? h[current_k][current_k] / gamma : 1.0;
                    s[current_k] = (gamma > 1e-12) ? h[current_k + 1][current_k] / gamma : 0.0;
                    h[current_k][current_k] = gamma;
                    g[current_k + 1] = -s[current_k] * g[current_k];
                    g[current_k] = c[current_k] * g[current_k];

                    if (fabs(g[current_k + 1]) < convergence_tol)
                    {
                        inner_completed = 1;
                        k_inner_final = current_k + 1;
                        break;
                    }
                }
            }

            if (!inner_completed)
                k_inner_final = gmres_iterations;
            total_iterations += k_inner_final;

            double *alpha = (double *)calloc(k_inner_final, sizeof(double));
            for (j = k_inner_final - 1; j >= 0; j--)
            {
                alpha[j] = g[j];
                for (ss = j + 1; ss < k_inner_final; ss++)
                {
                    alpha[j] -= h[j][ss] * alpha[ss];
                }
                if (fabs(h[j][j]) > 1e-12)
                    alpha[j] /= h[j][j];
            }
            for (j = 0; j < k_inner_final; j++)
            {
                for (ss = 0; ss < local_rows; ss++)
                {
                    x[ss] += alpha[j] * V[j][ss];
                }
            }
            free(alpha);

            Ax_vec = (double *)calloc(local_rows, sizeof(double));
            synchronizeVector(rank, num_procs, x, cntsent, to_be_sent, cntowner, receive, &communicator);
            sparseMatrixVectorMultiply(A, ja, ia, x, Ax_vec, local_rows);
            double local_true_res_sq = 0.0;
            for (i = 0; i < local_rows; i++)
            {
                double diff = b_original[i] - Ax_vec[i];
                local_true_res_sq += diff * diff;
            }
            free(Ax_vec);
            double global_true_res_sq;
            MPI_Allreduce(&local_true_res_sq, &global_true_res_sq, 1, MPI_DOUBLE, MPI_SUM, communicator);
            double true_residual = sqrt(global_true_res_sq);

            if (rank == 0)
            {
                printf("Info: Classical GMRES Restart %d, true residual norm after update = %e\n", rst, true_residual);
            }
            if (true_residual < true_residual_tol)
            {
                completed = 1;
            }
        }

        // Free classical GMRES memory in local scope
        for (i = 0; i <= gmres_iterations; i++)
        {
            free(V[i]);
            free(h[i]);
        }
        free(V);
        free(h);
        free(g);
        free(c);
        free(s);
    }

    t_end = MPI_Wtime();

    // Final result collection and printing
    double *full_solution = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        full_solution = (double *)malloc(global_rows * sizeof(double));
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));
        int base = global_rows / num_procs;
        int extra = global_rows % num_procs;
        int offset = 0;
        for (int i = 0; i < num_procs; i++)
        {
            recvcounts[i] = base + (i < extra ? 1 : 0);
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    MPI_Gatherv(x, local_rows, MPI_DOUBLE, full_solution, recvcounts, displs, MPI_DOUBLE, 0, communicator);

    g_comm_profile.total_time = t_end - t_start;

    // Output detailed communication performance analysis
    printDetailedPerformanceReport(rank, &communicator);

    if (rank == 0)
    {
        const char *algorithm_name = use_ca_gmres ? "CA-GMRES" : "Classical GMRES";
        if (completed)
            printf("\n%s algorithm converged successfully after %d total iterations.\n",
                   algorithm_name, total_iterations);
        else
            printf("\nWarning: %s algorithm did not converge within %d restarts.\n",
                   algorithm_name, max_restarts);
        // printf("Total execution time: %f seconds.\n", (t_end - t_start));
    }

    if (rank == 0)
    {
        // printDoubleArray("Final solution (full)", full_solution, global_rows);
        // Safe cleanup for rank 0 only
        if (full_solution)
        {
            free(full_solution);
            full_solution = NULL;
        }
        if (recvcounts)
        {
            free(recvcounts);
            recvcounts = NULL;
        }
        if (displs)
        {
            free(displs);
            displs = NULL;
        }
    }

    // Cleanup performance profiler
    cleanupCommunicationProfiler();

    // Simplified cleanup - let each algorithm handle its own memory
    //  Only cleanup the common resources
    if (rank == 0 && full_solution)
    {
        free(full_solution);
    }
    if (rank == 0)
    {
        if (recvcounts)
            free(recvcounts);
        if (displs)
            free(displs);
    }

    // Cleanup matrix and communication data
    free(A);
    free(ia);
    free(ja);
    free(x);
    free(X);
    if (M_inv)
        free(M_inv);
    if (b_original)
        free(b_original);

    // Cleanup communication arrays
    free(cntsent);
    free(cntowner);
    if (to_be_sent)
    {
        for (int i = 0; i < num_procs; ++i)
            if (to_be_sent[i])
                free(to_be_sent[i]);
        free(to_be_sent);
    }
    if (receive)
    {
        for (int i = 0; i < num_procs; ++i)
            if (receive[i])
                free(receive[i]);
        free(receive);
    }

    MPI_Finalize();
    return 0;
}
