#include <stdio.h>       // 用于 printf 函数
#include <stdlib.h>      // C标准库，在大型项目中通常包含以备不时之需
#include <mpi.h>         // 确保包含了 mpi.h
#include "sparse_blas.h" // 需要 parallelDotProduct
// 包含我们自己的头文件，确保实现与声明一致
#include "utility.h"

/**
 * @brief 打印程序的命令行用法说明。
 * 当用户提供错误或不足的参数时调用此函数。
 */
void printCommandSyntax()
{
    // 使用 fprintf(stderr, ...) 将错误信息输出到标准错误流，这是一种好的实践。
    // \n 是换行符，使输出更整洁。
    // 注意：这里的用法说明已根据 README.md 和 mpi_gmres.c 的实际情况更新。
    fprintf(stderr, "\n===============================================\n");
    fprintf(stderr, "Error: Not enough or incorrect input arguments.\n");
    fprintf(stderr, "===============================================\n\n");
    fprintf(stderr, "Usage: mpirun -np <processes> ./gmres <matrix_file> <rhs_file> <iterations> <preconditioner_flag> <restarts> [s_value]\n\n");
    fprintf(stderr, "Parameters:\n");
    fprintf(stderr, "  <processes>          - Number of MPI processes\n");
    fprintf(stderr, "  <matrix_file>        - Sparse matrix file in Matrix Market format\n");
    fprintf(stderr, "  <rhs_file>           - Right-hand side vector file\n");
    fprintf(stderr, "  <iterations>         - Maximum GMRES iterations per restart\n");
    fprintf(stderr, "  <preconditioner_flag>- 0 (disabled) or 1 (diagonal preconditioning)\n");
    fprintf(stderr, "  <restarts>           - Maximum number of restarts\n");
    fprintf(stderr, "  [s_value]            - Optional: s-step parameter (default=1)\n");
    fprintf(stderr, "                         s=1: Classical GMRES\n");
    fprintf(stderr, "                         s>1: CA-GMRES with s-step blocking\n\n");
}

/**
 * @brief 打印一个双精度浮点数矩阵。
 * @param name 矩阵的描述性名称，用于输出时标识。
 * @param matrixData 指向矩阵数据的指针 (按行主序存储)。
 * @param numRows 矩阵的行数。
 * @param numCols 矩阵的列数。
 */
void printDoubleMatrix(const char *name, const double *matrixData, int numRows, int numCols)
{
    printf("\n--- Matrix: %s ---\n", name);
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            // 通过行主序计算一维数组中的索引
            int index = i * numCols + j;

            // 使用 %14.7f 格式化输出，总宽度14，保留7位小数，更易于对齐和阅读
            printf("%14.7f", matrixData[index]);

            // 只要不是行尾的最后一个元素，就打印一个逗号和空格
            if (j < numCols - 1)
            {
                printf(", ");
            }
        }
        printf("\n"); // 每打印完一行后换行
    }
    printf("--- End of Matrix: %s ---\n\n", name);
}

/**
 * @brief 打印一个整数矩阵。
 * (函数结构与 printDoubleMatrix 非常相似)
 */
void printIntMatrix(const char *name, const int *matrixData, int numRows, int numCols)
{
    printf("\n--- Matrix: %s ---\n", name);
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            int index = i * numCols + j;
            // %8d 表示打印一个整数，宽度为8，使其对齐
            printf("%8d", matrixData[index]);
            if (j < numCols - 1)
            {
                printf(", ");
            }
        }
        printf("\n");
    }
    printf("--- End of Matrix: %s ---\n\n", name);
}

/**
 * @brief 打印一个双精度浮点数一维数组。
 * @param name 数组的描述性名称。
 * @param arrayData 指向数组数据的指针。
 * @param size 数组中的元素数量。
 */
void printDoubleArray(const char *name, const double *arrayData, int size)
{
    printf("\n--- Array: %s ---\n", name);
    for (int i = 0; i < size; i++)
    {
        // 打印每个元素的索引和值
        printf("%s[%d] = %f\n", name, i, arrayData[i]);
    }
    printf("--- End of Array: %s ---\n\n", name);
}

/**
 * @brief 打印一个整数一维数组。
 * (函数结构与 printDoubleArray 非常相似)
 */
void printIntArray(const char *name, const int *arrayData, int size)
{
    printf("\n--- Array: %s ---\n", name);
    for (int i = 0; i < size; i++)
    {
        printf("%s[%d] = %d\n", name, i, arrayData[i]);
    }
    printf("--- End of Array: %s ---\n\n", name);
}

void checkOrthogonality(const char *name, double **V, int num_vectors, int local_rows, int n_total, int rank, MPI_Comm *communicator)
{
    if (num_vectors <= 0)
        return;

    // 只有0号进程负责分配、计算和打印
    double *vtv_matrix = NULL;
    if (rank == 0)
    {
        vtv_matrix = (double *)malloc(num_vectors * num_vectors * sizeof(double));
        if (!vtv_matrix)
        {
            MPI_Abort(*communicator, 1);
        }
    }

    // 各进程协同计算 V[i]·V[j]
    for (int i = 0; i < num_vectors; i++)
    {
        for (int j = 0; j < num_vectors; j++)
        {
            double dot_product;
            parallelDotProduct(V[i], V[j], &dot_product, local_rows, communicator);

            if (rank == 0)
            {
                vtv_matrix[i * num_vectors + j] = dot_product;
            }
        }
    }

    // 0号进程打印结果
    if (rank == 0)
    {
        printDoubleMatrix(name, vtv_matrix, num_vectors, num_vectors);
        free(vtv_matrix);
    }
}