/*
 * par_gauss.c
 *
 * CS 470 Project 2 (OpenMP)
 * OpenMP parallelized version
 *
 * Compile with --std=c99
 */

#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// custom timing macros
#include "timer.h"

// uncomment this line to enable the alternative back substitution method
/*#define USE_COLUMN_BACKSUB*/

// use 64-bit IEEE arithmetic (change to "float" to use 32-bit arithmetic)
#define REAL double

// linear system: Ax = b    (A is n x n matrix; b and x are n x 1 vectors)
int n;
REAL *A;
REAL *x;
REAL *b;

// enable/disable debugging output (don't enable for large matrix sizes!)
bool debug_mode = false;

// enable/disable triangular mode (to skip the Gaussian elimination phase)
bool triangular_mode = false;

/*
 * Generate a random linear system of size n.
 */
void rand_system()
{
    // allocate space for matrices
    A = (REAL *)calloc(n * n, sizeof(REAL));
    b = (REAL *)calloc(n, sizeof(REAL));
    x = (REAL *)calloc(n, sizeof(REAL));

    // verify that memory allocation succeeded
    if (A == NULL || b == NULL || x == NULL)
    {
        printf("Unable to allocate memory for linear system\n");
        exit(EXIT_FAILURE);
    }

    // initialize pseudorandom number generator
    // (see https://en.wikipedia.org/wiki/Linear_congruential_generator)
    unsigned long seed = 0;

    // generate random matrix entries
    for (int row = 0; row < n; row++)
    {
        int col = triangular_mode ? row : 0;
        for (; col < n; col++)
        {
            if (row != col)
            {
                seed = (1103515245 * seed + 12345) % (1 << 31);
                A[row * n + col] = (REAL)seed / (REAL)ULONG_MAX;
            }
            else
            {
                A[row * n + col] = n / 10.0;
            }
        }
    }

    // generate right-hand side such that the solution matrix is all 1s
    for (int row = 0; row < n; row++)
    {
        b[row] = 0.0;
        for (int col = 0; col < n; col++)
        {
            b[row] += A[row * n + col] * 1.0;
        }
    }
}

/*
 * Reads a linear system of equations from a file in the form of an augmented
 * matrix [A][b].
 */
void read_system(const char *fn)
{
    // open file and read matrix dimensions
    FILE *fin = fopen(fn, "r");
    if (fin == NULL)
    {
        printf("Unable to open file \"%s\"\n", fn);
        exit(EXIT_FAILURE);
    }
    if (fscanf(fin, "%d\n", &n) != 1)
    {
        printf("Invalid matrix file format\n");
        exit(EXIT_FAILURE);
    }

    // allocate space for matrices
    A = (REAL *)malloc(sizeof(REAL) * n * n);
    b = (REAL *)malloc(sizeof(REAL) * n);
    x = (REAL *)malloc(sizeof(REAL) * n);

    // verify that memory allocation succeeded
    if (A == NULL || b == NULL || x == NULL)
    {
        printf("Unable to allocate memory for linear system\n");
        exit(EXIT_FAILURE);
    }

    // read all values
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (fscanf(fin, "%lf", &A[row * n + col]) != 1)
            {
                printf("Invalid matrix file format\n");
                exit(EXIT_FAILURE);
            }
        }
        if (fscanf(fin, "%lf", &b[row]) != 1)
        {
            printf("Invalid matrix file format\n");
            exit(EXIT_FAILURE);
        }
        x[row] = 0.0; // initialize x while we're reading A and b
    }
    fclose(fin);
}

/*
 * Performs Gaussian elimination on the linear system.
 * Assumes the matrix is singular and doesn't require any pivoting.
 */
void gaussian_elimination()
{
    // Outer loop is pivot selection, ie iterating over each column
    // Inherently sequential since each step depends on last
    for (int pivot = 0; pivot < n; pivot++)
    {
        // Loop in one indentation level iterates over the rows below the pivot
        // As pivot increases the remaining number of rows decreases, dynamic scheduling would allow for work load split, potentially specify block size (4?)
        // Since the coefficient computation and updating the matrix is independent from other rows parallel for works here
        // Each thread needs to be working on their own row, coeff, and col but they all need to be able to access
        // the matrix A, row b?, the pivot col, and n
        int row, col;
        REAL coeff;
#pragma omp parallel for default(none) shared(A, b, n, pivot) private(row, coeff, col) schedule(dynamic, 4)
        for (row = pivot + 1; row < n; row++)
        {
            coeff = A[row * n + pivot] / A[pivot * n + pivot];
            A[row * n + pivot] = 0.0;
            for (col = pivot + 1; col < n; col++)
            {
                A[row * n + col] -= A[pivot * n + col] * coeff;
            }
            b[row] -= b[pivot] * coeff;
        }
    }
}

/*
 * Performs backwards substitution on the linear system.
 * (row-oriented version)
 */
void back_substitution_row()
{
    REAL tmp;
    // The outer loop works its way up through the matrix and introduces a loop-carried dependency
    // Inherently sequential
    for (int row = n - 1; row >= 0; row--)
    {
        tmp = b[row];
        // Can compute tmp (dot product) parallel using a reduction to tmp
        // Each thread needs its own col but otherwise all variables can be shared
        // Could potentially add scheduling here as well (dyn) but I don't know how much that would improve
        int col;
#pragma omp parallel for default(none) shared(A, x, n, row) private(col) reduction(+ : tmp)
        for (col = row + 1; col < n; col++)
        {
            tmp += -A[row * n + col] * x[col];
        }
        x[row] = tmp / A[row * n + row];
    }
}

/*
 * Performs backwards substitution on the linear system.
 * (column-oriented version)
 */
void back_substitution_column()
{
    // Each row is set to b[row], this is clearly parallelizable

    // #pragma omp parallel for // default(none) shared(n, x, b)
    for (int row = 0; row < n; row++)
    {
        x[row] = b[row];
    }

    // Updating the rows can be parallelized per column which would be more effective that row orientation for larger matrices
    // HOWEVER it is a less cache efficient since the access pattern is column major and the matrices are stored row major, could lead to potential
    // cache thrashing as different threads try to access x[row] at the same time
    // n, x (solution vector), and A(coeff mat) all need to be share along with col, however each thread needs their own row

    // #pragma omp parallel for // default(none) shared(n, x, A, col) private(row)
    for (int col = n - 1; col >= 0; col--)
    {
        x[col] /= A[col * n + col];
        for (int row = 0; row < col; row++)
        {
            x[row] += -A[row * n + col] * x[col];
        }
    }
}

/*
 * Find the maximum error in the solution (only works for randomly-generated
 * matrices).
 */
REAL find_max_error()
{
    REAL error = 0.0, tmp;
    for (int row = 0; row < n; row++)
    {
        tmp = fabs(x[row] - 1.0);
        if (tmp > error)
        {
            error = tmp;
        }
    }
    return error;
}

/*
 * Prints a matrix to standard output in a fixed-width format.
 */
void print_matrix(REAL *mat, int rows, int cols)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            printf("%8.1e ", mat[row * cols + col]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    // check and parse command line options
    int c;
    while ((c = getopt(argc, argv, "dt")) != -1)
    {
        switch (c)
        {
        case 'd':
            debug_mode = true;
            break;
        case 't':
            triangular_mode = true;
            break;
        default:
            printf("Usage: %s [-dt] <file|size>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (optind != argc - 1)
    {
        printf("Usage: %s [-dt] <file|size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // read or generate linear system
    long int size = strtol(argv[optind], NULL, 10);
    START_TIMER(init)
    if (size == 0)
    {
        read_system(argv[optind]);
    }
    else
    {
        n = (int)size;
        rand_system();
    }
    STOP_TIMER(init)

    if (debug_mode)
    {
        printf("Original A = \n");
        print_matrix(A, n, n);
        printf("Original b = \n");
        print_matrix(b, n, 1);
    }

    // perform gaussian elimination
    START_TIMER(gaus)
    if (!triangular_mode)
    {
        gaussian_elimination();
    }
    STOP_TIMER(gaus)

    // perform backwards substitution
    START_TIMER(bsub)
#ifndef USE_COLUMN_BACKSUB
    back_substitution_row();
#else
    back_substitution_column();
#endif
    STOP_TIMER(bsub)

    if (debug_mode)
    {
        printf("Triangular A = \n");
        print_matrix(A, n, n);
        printf("Updated b = \n");
        print_matrix(b, n, 1);
        printf("Solution x = \n");
        print_matrix(x, n, 1);
    }

    // print results
    printf("Nthreads=%2d  ERR=%8.1e  INIT: %8.4fs  GAUS: %8.4fs  BSUB: %8.4fs\n",
           1, find_max_error(),
           GET_TIMER(init), GET_TIMER(gaus), GET_TIMER(bsub));

    // clean up and exit
    free(A);
    free(b);
    free(x);
    return EXIT_SUCCESS;
}
