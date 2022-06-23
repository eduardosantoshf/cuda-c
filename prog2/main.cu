#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "../common/common.h"
#include "utils/utils.cuh"

#ifndef SECTOR_SIZE
# define SECTOR_SIZE  512
#endif

#ifndef N_SECTORS
# define N_SECTORS    (1 << 21)
#endif

/**
 * @brief Host processing logic, row by row.
 */
void hostRR(int order, int amount, double **matrixArray, double *results);

/**
 * @brief Device processing logic, row by row.
 */
__global__ void deviceRR(double *d_matrixArray, double *d_results);

// process the called command
static int processCommand(int argc, char *argv[], int* , char*** fileNames);

// print the explanation of how to use the command
static void printUsage(char *cmdName);

/**
 * @brief Main logic of the program.
 * Makes gaussian elimination on the host and the device.
 * Compares thre obtained results at the end.
 *
 * @param argc amount of arguments in the command line
 * @param argv array with the arguments from the command line
 * @return int return execution status of operation
 */
int main(int argc, char **argv)
{
    // process command line information to obtain file names
    int fileAmount = 0;
    char ** fileNames;
    int command_result;

    // process the command and act according to it
    command_result = process_command(argc, argv, &fileAmount, &fileNames);

    if (command_result != EXIT_SUCCESS) {
        perror("There was an error processing the input");
        exit(EXIT_FAILURE);
    }

    // device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // process files
    double *h_matrixArray = NULL;
    int order = 0, amount = 0;
    for(int i = 0; i < fileAmount; i++)
    {
        // read data from file
        readData(*(fileNames + i), &h_matrixArray, &order, &amount);

        // structure to save results
        double *retrieved_results = (double *)malloc(sizeof(double) * amount);

        // allocate memory on device
        double *d_matrixArray;
        double *d_results;
        CHECK(cudaMalloc((void **)&d_matrixArray, (sizeof(double) * order * order * amount)));
        CHECK(cudaMalloc((void **)&d_results, sizeof(double) * amount));

        // copy data to device memory
        CHECK(cudaMemcpy(d_matrixArray, h_matrixArray, (sizeof(double) * order * order * amount), cudaMemcpyHostToDevice));

        // create grid and block
        dim3 grid(amount, 1, 1);
        dim3 block(order, 1, 1);

        // DEVICE PROCESSING
        double d_start = seconds();
        deviceRR<<<grid, block>>>(d_matrixArray, d_results);
        CHECK (cudaDeviceSynchronize ());
        double drr = seconds() - d_start;

        CHECK(cudaGetLastError ());         // check kernel errors
        CHECK(cudaMemcpy(retrieved_results, d_results, sizeof(double) * amount, cudaMemcpyDeviceToHost));   // return obtained results
        CHECK(cudaFree (d_matrixArray));    // free device memory

        // HOST PROCESSING
        double h_results[amount];
        double start = seconds();
        hostRR(order, amount, &h_matrixArray, h_results);
        double hrr = seconds() - start;

        printf("\nRESULTS\n");
        for(int i = 0; i < amount; i++)
        {
            printf("MATRIX: <%d>\tHOST: <%+5.3e>\t DEVICE: <%+5.3e>\n", i + 1, h_results[i], retrieved_results[i]);
        }

        printf("\nEXECUTION TIMES\n");
        printf("Host processing took <%.5f> seconds.\n", hrr);
        printf("Device processing took <%.5f> seconds.\n", drr);
    }

    return 0;
}

/**
 * @brief Calculates determinant row by row
 *
 * @param matrix pointer to matrix
 * @param order order of matrix
 */
void hostRR(int order, int amount, double **matrixArray, double *results)
{
    for(int i = 0; i < amount; i++)
    {
        *(results + i) = row_by_row_determinant(order, ((*matrixArray) + (i * order * order)));
        // printf("%+5.3e\n", *(results + i));
    }
}

/**
 * @brief Device kernel to calculate gaussian elimination, row by row
 *
 * @param d_matrixArray pointer to array of matrices
 * @param d_results pointer to array of results
 */
__global__ void deviceRR(double *d_matrixArray, double *d_results)
{
    int n = blockDim.x;

    for(int iter = 0; iter < n; iter++)
    {
        if(threadIdx.x < iter)
            continue;

        int matrixIdx = blockIdx.x * n * n;
        int row = matrixIdx + threadIdx.x * n;
        int iterRow = matrixIdx + iter * n;

        if(threadIdx.x == iter)
        {
            if(iter == 0)
            {
                *(d_results + blockIdx.x) = 1;
            }
            *(d_results + blockIdx.x) *= *(d_matrixArray + iterRow + iter);
            continue;
        }

        double pivot = *(d_matrixArray + iterRow + iter);

        double value = *(d_matrixArray + row + iter) / pivot;
        for(int i = iter + 1; i < n; i++)
        {
            *(d_matrixArray + row + i) -= *(d_matrixArray + iterRow + i) * value;
        }
        __syncthreads();
    }
}

/**
 * @brief Process command line input
 *
 * Iterates through argv to find and store thread amount, file amount and file names.
 *
 * @param argc Argument quantity in the command line
 * @param argv Array with arguments fromt he command line
 * @param fileAmount Pointer to file amount
 * @param fileNames Pointer to pointer array where file names are stored
 * @return int Return value of command line processing
 */
int processCommand(int argc, char *argv[], int* fileAmount, char*** fileNames)
{

    char **auxFileNames = NULL;
    int opt;    // selected option

    if(argc <= 2)
    {
        perror("No/few arguments were provided.");
        printUsage(basename(strdup("PROGRAM")));
        return EXIT_FAILURE;
    }

    opterr = 0;
    do
    {
        switch ((opt = getopt (argc, argv, "f:h")))
        {
            case 'f':
            {                                                // case: file name
                if (optarg[0] == '-')
                {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename (argv[0]));
                    return EXIT_FAILURE;
                }

                int index = optind - 1;
                char *next = NULL;

                while(index < argc)
                {
                    next = argv[index++];                               // get next element in argv

                    if(next[0] != '-')                                  // if element isn't an option, then its a file name
                    {
                        if((*fileAmount) == 0)                          // first file name
                        {
                            auxFileNames = (char **)malloc(sizeof(char*) * (++(*fileAmount)));
                            if(!auxFileNames)                           // error reallocating memory
                            {
                                fprintf(stderr, "error allocating memory for file name\n");
                                return EXIT_FAILURE;
                            }
                            *(auxFileNames + (*fileAmount) - 1) = next;
                        }
                        else                                            // following file names
                        {
                            (*fileAmount)++;
                            auxFileNames = (char **)realloc(auxFileNames, sizeof(char*) * (*fileAmount));
                            if(!auxFileNames)                           // error reallocating memory
                            {
                                fprintf(stderr, "error reallocating memory for file name\n");
                                return EXIT_FAILURE;
                            }
                            *(auxFileNames + (*fileAmount) -1) = next;
                        }
                    }
                    else                                                // element is something else
                        break;
                }
                break;
            }

            case 'h':                                                   // case: help mode
            {
                printUsage (basename (argv[0]));
                return EXIT_SUCCESS;
            }

            case '?':                                                   // case: invalid option
            {
                fprintf(stderr, "%s: invalid option\n", basename (argv[0]));
                printUsage(basename (argv[0]));
                return EXIT_FAILURE;
            }

            default:
                break;
        }

    } while (opt != -1);

    // print file names
    printf("File amount: <%d>\nFile names:\n", (*fileAmount));
    for(int i = 0; i < (*fileAmount); i++)
    {
        char* nome = *(auxFileNames + i);
        printf("\tfile: <%s>\n", nome);
    }

    // copy auxiliar pointer to fileNames pointer
    *fileNames = auxFileNames;

    return EXIT_SUCCESS;

}

/**
 *  @brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  @param cmdName string with the name of the command
 */
static void printUsage(char *cmdName)
{
    fprintf (stderr,
        "\nSynopsis: %s OPTIONS [filename / positive number]\n"
        "  OPTIONS:\n"
        "  -h      --- print this help\n"
        "  -f      --- filename\n"
        "  -n      --- positive number\n", cmdName);
}