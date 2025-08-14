#include "localization.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "communication_profiler.h"

int setupCommunication(int rank, int num_procs, int global_rows, int *col_indices, const int *row_ptr, int local_non_zeros, MPI_Comm *communicator, int **send_counts, int ***send_lists, int **recv_counts, int ***recv_lists)
{
    // Determine row distribution for ALL processes
    int *counts = (int *)malloc(num_procs * sizeof(int));
    int *displs = (int *)malloc(num_procs * sizeof(int));
    int base_rows = global_rows / num_procs;
    int extra_rows = global_rows % num_procs;
    int offset = 0;
    for (int i = 0; i < num_procs; i++)
    {
        counts[i] = base_rows + (i < extra_rows ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }

    const int local_rows = counts[rank];
    int next_local_idx = local_rows;
    int *tag = (int *)calloc(global_rows, sizeof(int));

    int **expected_from_owner = (int **)malloc(num_procs * sizeof(int *));
    *recv_lists = (int **)malloc(num_procs * sizeof(int *));
    *send_lists = (int **)malloc(num_procs * sizeof(int *));

    *recv_counts = (int *)calloc(num_procs, sizeof(int));
    *send_counts = (int *)calloc(num_procs, sizeof(int));

    for (int i = 0; i < num_procs; i++)
    {
        expected_from_owner[i] = NULL;
        (*recv_lists)[i] = NULL;
        (*send_lists)[i] = NULL;
    }

    // Identify required non-local columns
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            const int global_col = col_indices[j];

            // Correctly find the owner rank
            int owner_rank = -1;
            for (int p = 0; p < num_procs; ++p)
            {
                if (global_col >= displs[p] && global_col < displs[p] + counts[p])
                {
                    owner_rank = p;
                    break;
                }
            }

            if (owner_rank == rank)
            {
                // Convert to local index for this process
                col_indices[j] = global_col - displs[rank];
            }
            else
            {
                if (tag[global_col] == 0)
                {
                    const int new_local_idx = next_local_idx++;
                    tag[global_col] = new_local_idx;

                    (*recv_counts)[owner_rank]++;
                    (*recv_lists)[owner_rank] = (int *)realloc((*recv_lists)[owner_rank], (*recv_counts)[owner_rank] * sizeof(int));
                    expected_from_owner[owner_rank] = (int *)realloc(expected_from_owner[owner_rank], (*recv_counts)[owner_rank] * sizeof(int));

                    (*recv_lists)[owner_rank][(*recv_counts)[owner_rank] - 1] = new_local_idx;
                    expected_from_owner[owner_rank][(*recv_counts)[owner_rank] - 1] = global_col;
                }
                col_indices[j] = tag[global_col];
            }
        }
    }

    // Exchange communication requirements with other processes
    MPI_Request requests[2 * num_procs];
    int req_count = 0;

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if ((*recv_counts)[p] > 0)
        {
            MPI_Isend(&(*recv_counts)[p], 1, MPI_INT, p, 0, *communicator, &requests[req_count++]);
            MPI_Isend(expected_from_owner[p], (*recv_counts)[p], MPI_INT, p, 1, *communicator, &requests[req_count++]);
        }
        else
        { // Still need to send a zero count
            MPI_Isend(&(*recv_counts)[p], 1, MPI_INT, p, 0, *communicator, &requests[req_count++]);
        }
    }

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        MPI_Status status;
        MPI_Recv(&(*send_counts)[p], 1, MPI_INT, p, 0, *communicator, &status);
        if ((*send_counts)[p] > 0)
        {
            (*send_lists)[p] = (int *)malloc((*send_counts)[p] * sizeof(int));
            MPI_Recv((*send_lists)[p], (*send_counts)[p], MPI_INT, p, 1, *communicator, &status);
            // Correctly convert received global indices to local indices for this process
            for (int i = 0; i < (*send_counts)[p]; i++)
            {
                (*send_lists)[p][i] -= displs[rank];
            }
        }
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // Step 4: Cleanup
    for (int i = 0; i < num_procs; i++)
    {
        if (expected_from_owner[i])
            free(expected_from_owner[i]);
    }
    free(expected_from_owner);
    free(tag);
    free(counts);
    free(displs);

    return 0;
}

int synchronizeVector1(int rank, int num_procs, double *vector_data,
                       const int *send_counts, int **const send_lists,
                       const int *recv_counts, int **const recv_lists,
                       MPI_Comm *communicator)
{

    MPI_Request requests[(num_procs - 1) * 2];
    int request_count = 0;

    double **send_buffers = (double **)malloc(num_procs * sizeof(double *));
    double **recv_buffers = (double **)malloc(num_procs * sizeof(double *));

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (recv_counts[p] > 0)
        {
            recv_buffers[p] = (double *)malloc(recv_counts[p] * sizeof(double));
            MPI_Irecv(recv_buffers[p], recv_counts[p], MPI_DOUBLE, p, 100, *communicator, &requests[request_count++]);
        }
    }

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (send_counts[p] > 0)
        {
            send_buffers[p] = (double *)malloc(send_counts[p] * sizeof(double));
            for (int i = 0; i < send_counts[p]; i++)
            {
                send_buffers[p][i] = vector_data[send_lists[p][i]];
            }
            MPI_Isend(send_buffers[p], send_counts[p], MPI_DOUBLE, p, 100, *communicator, &requests[request_count++]);
        }
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (recv_counts[p] > 0)
        {
            for (int i = 0; i < recv_counts[p]; i++)
            {
                vector_data[recv_lists[p][i]] = recv_buffers[p][i];
            }
        }
    }

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (recv_counts[p] > 0)
            free(recv_buffers[p]);
        if (send_counts[p] > 0)
            free(send_buffers[p]);
    }
    free(recv_buffers);
    free(send_buffers);

    return 0;
}

int synchronizeVector(int rank, int num_procs, double *vector_data,
                      const int *send_counts, int **const send_lists,
                      const int *recv_counts, int **const recv_lists,
                      MPI_Comm *communicator)
{

    double sync_start = MPI_Wtime();

    MPI_Request requests[(num_procs - 1) * 2];
    int request_count = 0;

    double **send_buffers = (double **)malloc(num_procs * sizeof(double *));
    double **recv_buffers = (double **)malloc(num_procs * sizeof(double *));

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (recv_counts[p] > 0)
        {
            recv_buffers[p] = (double *)malloc(recv_counts[p] * sizeof(double));
            MPI_Irecv(recv_buffers[p], recv_counts[p], MPI_DOUBLE, p, 100, *communicator, &requests[request_count++]);
        }
    }

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (send_counts[p] > 0)
        {
            send_buffers[p] = (double *)malloc(send_counts[p] * sizeof(double));
            for (int i = 0; i < send_counts[p]; i++)
            {
                send_buffers[p][i] = vector_data[send_lists[p][i]];
            }
            MPI_Isend(send_buffers[p], send_counts[p], MPI_DOUBLE, p, 100, *communicator, &requests[request_count++]);
        }
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (recv_counts[p] > 0)
        {
            for (int i = 0; i < recv_counts[p]; i++)
            {
                vector_data[recv_lists[p][i]] = recv_buffers[p][i];
            }
        }
    }

    for (int p = 0; p < num_procs; p++)
    {
        if (p == rank)
            continue;
        if (recv_counts[p] > 0)
            free(recv_buffers[p]);
        if (send_counts[p] > 0)
            free(send_buffers[p]);
    }
    free(recv_buffers);
    free(send_buffers);
    double sync_end = MPI_Wtime();
    recordVectorSynchronization(sync_end - sync_start);

    return 0;
}

int printCommunicationPlan(int rank, int num_procs, const int *send_counts, int **const send_lists, const int *recv_counts, int **const recv_lists, MPI_Comm *communicator)
{

    for (int p = 0; p < num_procs; p++)
    {
        MPI_Barrier(*communicator);
        if (p == rank)
        {
            printf("\n--- Communication Plan for Process %d ---\n", rank);
            for (int i = 0; i < num_procs; i++)
            {
                if (i == rank || recv_counts[i] == 0)
                    continue;
                printf("    Will receive %d elements from process %d.\n", recv_counts[i], i);
            }
            for (int i = 0; i < num_procs; i++)
            {
                if (i == rank || send_counts[i] == 0)
                    continue;
                printf("    Will send %d elements to process %d.\n", send_counts[i], i);
            }
            printf("--- End of Plan for Process %d ---\n", rank);
        }
    }
    return 0;
}
