#ifndef UTILITY_H
#define UTILITY_H

// 打印程序如何运行的命令行语法提示
void printCommandSyntax();

// 打印一个双精度浮点数矩阵
// name: 矩阵的描述性名称
// matrixData: 指向矩阵数据的指针 (按行主序存储)
// numRows: 矩阵的行数
// numCols: 矩阵的列数
void printDoubleMatrix(const char *name, const double *matrixData, int numRows, int numCols);

// 打印一个整数矩阵
void printIntMatrix(const char *name, const int *matrixData, int numRows, int numCols);

// 打印一个双精度浮点数数组
// name: 数组的描述性名称
// arrayData: 指向数组数据的指针
// size: 数组中的元素数量
void printDoubleArray(const char *name, const double *arrayData, int size);

// 打印一个整数数组
void printIntArray(const char *name, const int *arrayData, int size);

/**
 * @brief 检查一组分布式向量的正交性，由0号进程打印 V^T * V 矩阵。
 * @param name 描述性名称
 * @param V 存储向量的数组
 * @param num_vectors 要检查的向量数量
 * @param local_rows 每个进程的向量行数
 * @param n_total 向量总维度（含幽灵点）
 * @param rank 当前进程号
 * @param communicator MPI通信域
 */
void checkOrthogonality(const char *name, double **V, int num_vectors, int local_rows, int n_total, int rank, MPI_Comm *communicator);
#endif // UTILITY_H