#ifndef COMMUNICATION_PROFILER_H
#define COMMUNICATION_PROFILER_H
#include <mpi.h>

// Communication performance statistics structure
typedef struct {
   // Overall statistics
   double total_time;
   double computation_time;
   double communication_time;
   
   // Inner product statistics
   int inner_product_count;
   double inner_product_time;
   
   // Vector synchronization statistics
   int vector_sync_count;
   double vector_sync_time;
   
   // Other statistics
   int total_iterations;
} CommunicationProfile;

// Global performance profiler
extern CommunicationProfile g_comm_profile;

/**
* @brief Initialize communication performance profiler
*/
void initCommunicationProfiler(int rank, int num_procs);

/**
* @brief Record communication overhead of inner product operations
*/
void recordInnerProductCommunication(double comm_time);

/**
* @brief Record communication overhead of vector synchronization
*/
void recordVectorSynchronization(double sync_time);

/**
* @brief Record iteration information
*/
void recordIteration();

/**
* @brief Record computation time
*/
void recordComputation(double comp_time);

/**
* @brief Print detailed communication performance report
*/
void printDetailedPerformanceReport(int rank, MPI_Comm *communicator);

/**
* @brief Clean up performance profiler resources
*/
void cleanupCommunicationProfiler();

/**
* @brief Record bandwidth usage for data transfers
*/
void recordBandwidthUsage(double data_size, double transfer_time);

/**
* @brief Calculate and display efficiency metrics
*/
void calculateEfficiencyMetrics(int rank, int num_procs, MPI_Comm *communicator);

/**
* @brief Export performance data to CSV format
*/
void exportPerformanceDataCSV(const char *filename, int rank);

/**
* @brief Get current performance statistics
*/
CommunicationProfile* getCurrentProfile();

#endif // COMMUNICATION_PROFILER_H