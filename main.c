#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 5

void print_matrix1(int rank, int n, float *matrix)
{
    int i, j;
    printf("---------------%d-----------------\n", rank);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(int rank, int n, float v[])
{
    int i;
    printf("-------%d---------\n", rank);
    for (i = 0; i < n; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n\n");
}

void print_matrix2(int rank, int n, float matrix[][N])
{
    int i, j;
    printf("---------------%d-----------------\n", rank);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{

    int i, j, rank, numprocs, filas, filas_extra;
    float *matrix, *matrixAux;
    float vector[N];
    float *result, *resultAux;
    struct timeval tv1, tv2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    filas = ceil(N / numprocs);
    filas_extra = (N % numprocs != 0 && rank == numprocs - 1) ? (numprocs * filas - N) : 0;

    printf("%d-----%d\n", filas, filas_extra);

    if ((matrixAux = malloc(sizeof(float) * (filas + filas_extra) * N)) == NULL)
        perror("error 1: ");
    if ((resultAux = malloc(sizeof(float) * (filas + filas_extra))) == NULL)
        perror("error 2: ");

    /* Initialize Matrix and Vector */
    if (rank == 0)
    {
        if ((matrix = malloc(sizeof(float) * N * N)) == NULL)
            perror("error 3: ");
        if ((result = malloc(sizeof(float) * N)) == NULL)
            perror("error 4: ");

        for (int i = 0; i < N; i++)
        {
            vector[i] = i;

            for (int j = 0; j < N; j++)
            {
                matrix[i * N + j] = i + j;
            }
        }
        print_matrix1(rank, N, matrix);
    }
    MPI_Scatter(matrix, N * filas, MPI_FLOAT, matrixAux, N * filas, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank != 2)
        print_matrix1(rank, filas, matrixAux);
    else
        print_matrix1(rank, filas + (numprocs * filas - N), matrixAux);

    // COMUNICADO ???
    gettimeofday(&tv1, NULL);

    for (i = 0; i < filas; i++)
    {

        resultAux[i] = 0;
        for (j = 0; j < N; j++)
        {
            resultAux[i] += matrixAux[i * N + j] * vector[j];
        }
    }

    gettimeofday(&tv2, NULL);

    int microseconds = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    MPI_Gather(resultAux, filas, MPI_FLOAT, result, filas, MPI_FLOAT, 0, MPI_COMM_WORLD);
    printf("Time (seconds) = %lf\n", (double)microseconds / 1E6);

    /*Display result */

    if (rank == 0)
    {
        for (i = 0; i < N; i++)
        {
            printf(" %f \t ", result[i]);
        }
        free(matrix);
        free(result);
    }

    free(matrixAux);
    free(resultAux);

    MPI_Finalize();

    return 0;
}
