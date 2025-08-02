#include "communication_profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

CommunicationProfile g_comm_profile;
static int g_rank = 0;
static int g_num_procs = 0;

void initCommunicationProfiler(int rank, int num_procs)
{
    g_rank = rank;
    g_num_procs = num_procs;

    memset(&g_comm_profile, 0, sizeof(CommunicationProfile));

    if (rank == 0)
    {
        printf("\n=== Communication Profiler Initialized ===\n");
        printf("Processes: %d\n", num_procs);
        printf("==========================================\n\n");
    }
}

void recordInnerProductCommunication(double comm_time)
{
    g_comm_profile.inner_product_count++;
    g_comm_profile.inner_product_time += comm_time;
    g_comm_profile.communication_time += comm_time;
}

void recordVectorSynchronization(double sync_time)
{
    g_comm_profile.vector_sync_count++;
    g_comm_profile.vector_sync_time += sync_time;
    g_comm_profile.communication_time += sync_time;
}

void recordIteration()
{
    g_comm_profile.total_iterations++;
}

void recordComputation(double comp_time)
{
    g_comm_profile.computation_time += comp_time;
}

void printDetailedPerformanceReport(int rank, MPI_Comm *communicator)
{

    double global_comm_time, global_comp_time;
    int global_inner_product_count, global_vector_sync_count;
    int global_total_iterations;

    MPI_Allreduce(&g_comm_profile.communication_time, &global_comm_time, 1, MPI_DOUBLE, MPI_SUM, *communicator);
    MPI_Allreduce(&g_comm_profile.computation_time, &global_comp_time, 1, MPI_DOUBLE, MPI_SUM, *communicator);
    MPI_Allreduce(&g_comm_profile.inner_product_count, &global_inner_product_count, 1, MPI_INT, MPI_MAX, *communicator);
    MPI_Allreduce(&g_comm_profile.vector_sync_count, &global_vector_sync_count, 1, MPI_INT, MPI_MAX, *communicator);
    MPI_Allreduce(&g_comm_profile.total_iterations, &global_total_iterations, 1, MPI_INT, MPI_MAX, *communicator);

    global_comm_time /= g_num_procs;
    global_comp_time /= g_num_procs;
    double total_time = global_comm_time + global_comp_time;

    if (rank == 0)
    {
        printf("\n============================================================\n");
        printf("           Task 3: Initial Communication Profiling\n");
        printf("\n============================================================\n");

        printf("\n--- OVERALL PERFORMANCE ---\n");
        printf("Total Execution Time:     %.6f seconds\n", total_time);
        printf("Total Communication Time: %.6f seconds (%.1f%%)\n",
               global_comm_time, (global_comm_time / total_time) * 100);
        printf("Total Computation Time:   %.6f seconds (%.1f%%)\n",
               global_comp_time, (global_comp_time / total_time) * 100);

        printf("\n--- INNER PRODUCT ANALYSIS ---\n");
        printf("Inner Product Operations: %d\n", global_inner_product_count);
        printf("Inner Product Comm Time:  %.6f seconds\n", g_comm_profile.inner_product_time);
        if (global_inner_product_count > 0)
        {
            double avg_inner_product_time = (g_comm_profile.inner_product_time / global_inner_product_count) * 1000;
            printf("Average Inner Product Time: %.3f ms\n", avg_inner_product_time);
        }
        printf("Inner Product Comm Ratio: %.1f%% of total comm\n",
               (g_comm_profile.inner_product_time / global_comm_time) * 100);

        printf("\n--- VECTOR SYNCHRONIZATION ANALYSIS ---\n");
        printf("Vector Sync Operations:   %d\n", global_vector_sync_count);
        printf("Vector Sync Comm Time:    %.6f seconds\n", g_comm_profile.vector_sync_time);
        if (global_vector_sync_count > 0)
        {
            double avg_sync_time = (g_comm_profile.vector_sync_time / global_vector_sync_count) * 1000;
            printf("Average Sync Time:        %.3f ms\n", avg_sync_time);
        }

        printf("\n--- ALGORITHM STATISTICS ---\n");
        printf("Total GMRES Iterations:   %d\n", global_total_iterations);
        if (global_total_iterations > 0)
        {
            printf("Avg Inner Products/Iter:  %.1f\n", (double)global_inner_product_count / global_total_iterations);
        }

        printf("\n--- PERFORMANCE TABLE ---\n");
        printf("Processes | Total Time | Comm Time | Comm %% | Inner Prod | Avg IP Time\n");
        printf("----------|------------|-----------|--------|------------|------------\n");
        printf("%9d | %10.6f | %9.6f | %6.1f | %10d | %8.3f ms\n",
               g_num_procs, total_time, global_comm_time,
               (global_comm_time / total_time) * 100, global_inner_product_count,
               global_inner_product_count > 0 ? (g_comm_profile.inner_product_time / global_inner_product_count) * 1000 : 0.0);

        printf("\n============================================================\n");
    }
}

void cleanupCommunicationProfiler()
{

    memset(&g_comm_profile, 0, sizeof(CommunicationProfile));
}