#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <libgen.h>

#include "../common/common.h"
#include "computeDeterminant.cuh"

/**
 * @brief Host processing logic
 */
void computeDeterminantHost(int order, int amount, double **matrix, double *results);

/**
 * @brief GPU processing logic
 */
__global__ void computeDeterminantGPU(double *deviceMatrix, double *deviceResults);

/**
 * @brief Read data from files
 */
void fileHandling(char *fileName, double **matrixArray, int *order, int *amount);

/**
 * @brief Process the called command
 */
static int processCommand(int argc, char *argv[], int* , char*** fileNames);

/**
 * @brief Print the explanation of how to use the command
 */
static void printUsage(char *cmdName);

/**
 * @brief Main logic of the solution
 *
 * @param argc amount arguments in the command line
 * @param argv arguments from the command line
 * @return int execution status of operation
 */
int main(int argc, char **argv) {
    // process command line information to obtain file names
    int fileCount = 0;
    char ** fileNames;
    int commandResult;

    // process the command and act according to it
    commandResult = processCommand(argc, argv, &fileCount, &fileNames);

    if (commandResult != EXIT_SUCCESS) {
        perror("ERROR: there was an error processing the input!");
        exit(EXIT_FAILURE);
    }

    // device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nUsing device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // process files
    double *hostMatrix = NULL;
    int order = 0, amount = 0;
    
    for (int i = 0; i < fileCount; i++) {
        // read data from file
        fileHandling(*(fileNames + i), &hostMatrix, &order, &amount);

        // structure to save results
        double *retrievedResults = (double *)malloc(sizeof(double) * amount);

        double *deviceMatrix;
        double *deviceResults;
        
        // allocate memory
        CHECK(cudaMalloc((void **)&deviceMatrix, (sizeof(double) * order * order * amount)));
        CHECK(cudaMalloc((void **)&deviceResults, sizeof(double) * amount));

        // copy data to device memory
        CHECK(cudaMemcpy(deviceMatrix, hostMatrix, (sizeof(double) * order * order * amount), cudaMemcpyHostToDevice));

        // create grid and block
        dim3 grid(amount); // by omission, grid(amount) <=> grid(amount, 1, 1)
        dim3 block(order); // by omission, block(order) <=> block(order, 1, 1)

        // process on device
        double deviceStart = seconds();

        computeDeterminantGPU<<<grid, block>>>(deviceMatrix, deviceResults);
        CHECK(cudaDeviceSynchronize());

        double deviceTime = seconds() - deviceStart;

        CHECK(cudaGetLastError()); // check kernel errors
        CHECK(cudaMemcpy(retrievedResults, deviceResults, sizeof(double) * amount, cudaMemcpyDeviceToHost)); // return results
        CHECK(cudaFree (deviceMatrix)); // free device memory

        // process on host
        double hostResults[amount];
        double hostStart = seconds();

        computeDeterminantHost(order, amount, &hostMatrix, hostResults);

        double hostTime = seconds() - hostStart;

        printf("\nResults:\n");

        for(int i = 0; i < amount; i++) {
            printf("\nMatrix nº %d", i + 1);
            printf("\nDeterminant on Host: %+5.3e", hostResults[i]);
            printf("\nDeterminant on Device: %+5.3e\n", retrievedResults[i]);
        }

        printf("\nHost processing time: %.5f s\n", hostTime);
        printf("Device processing time: %.5f s\n", deviceTime);
    }

    return 0;
}

/**
 * @brief Computes the determinant row by row on the host
 *
 * @param matrix pointer to matrix
 * @param order order of matrix
 */
void computeDeterminantHost(int order, int amount, double **matrix, double *results) {
    for (int i = 0; i < amount; i++)
        *(results + i) = computeDeterminant(order, ((*matrix) + (i * order * order)));
}

/**
 * @brief Computes the determinant row by row on the GPU
 *
 * @param deviceMatrix matrices array
 * @param deviceResults results array
 */
__global__ void computeDeterminantGPU(double *deviceMatrix, double *deviceResults) {
    int n = blockDim.x; // order of a matrix (= size of the block, because a block is a matrix)

	for (int iteration = 0; iteration < n; iteration++) {
        if (threadIdx.x < iteration) 
            continue;

        int matrixID = blockIdx.x * n * n; // jump to the current matrix 
        int row = matrixID + threadIdx.x * n; // current row offset of this (block thread)	
        int iterationRow = matrixID + iteration * n; // current iteration

        if (threadIdx.x == iteration) { // thread does the partial pivoting, updating the determinant value of the matrix
            if (iteration == 0)
                deviceResults[blockIdx.x] = 1; // initialize the results

            deviceResults[blockIdx.x] *= deviceMatrix[iterationRow + iteration];

            continue;
        }

        double pivot = deviceMatrix[iterationRow + iteration];

        double value = deviceMatrix[row + iteration] / pivot;

        for (int i = iteration + 1; i < n; i++)
            deviceMatrix[row + i] -= deviceMatrix[iterationRow + i] * value; 

        __syncthreads();
    }
}

/**
 * @brief Read matrices data from a file
 *
 * @param fileName filename
 * @param matrixArray matrices array
 * @param order matrices order
 * @param amount number of matrices
 */
void fileHandling(char *fileName, double **matrixArray, int *order, int *amount) {
    FILE *f = fopen(fileName, "rb");
    
    if (!f) {
        perror("ERROR: can't open file!");
        exit(EXIT_FAILURE);
    }

    fread(amount, sizeof(int), 1, f); // read matrices amount

    fread(order, sizeof(int), 1, f); // read matrices order

    (*matrixArray) = (double*)malloc(sizeof(double) * (*amount) * (*order) * (*order));

    fread((*matrixArray), sizeof(double), (*amount) * (*order) * (*order), f); // read all matrices
}

/**
 * @brief Process command line input
 *
 * Iterates through argv to find and store thread amount, file amount and file names.
 *
 * @param argc Argument quantity in the command line
 * @param argv Array with arguments fromt he command line
 * @param fileCount Pointer to file amount
 * @param fileNames Pointer to pointer array where file names are stored
 * @return int Return value of command line processing
 */
int processCommand(int argc, char *argv[], int* fileCount, char*** fileNames) {
    char **tempFilenames = NULL;
    int opt;    // selected option

    if(argc <= 2) {
        perror("No/few arguments were provided.");
        printUsage(basename(strdup("PROGRAM")));
        return EXIT_FAILURE;
    }

    opterr = 0;
    do {
        switch ((opt = getopt (argc, argv, "f:h"))) {
            case 'f':
            {                                                
                if (optarg[0] == '-') {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename (argv[0]));
                    return EXIT_FAILURE;
                }

                int index = optind - 1;
                char *next = NULL;

                while(index < argc) {
                    // next arg
                    next = argv[index++];                               

                    if((*fileCount) == 0) {
                        tempFilenames = (char **)malloc(sizeof(char*) * (++(*fileCount)));
                        *(tempFilenames + (*fileCount) - 1) = next;
                    } else {
                        (*fileCount)++;
                        tempFilenames = (char **)realloc(tempFilenames, sizeof(char*) * (*fileCount));
                        *(tempFilenames + (*fileCount) -1) = next;
                    }

                }
                break;
            }

            // help
            case 'h':                                                   
            {
                printUsage (basename (argv[0]));
                return EXIT_SUCCESS;
            }
            // invalid
            case '?':                                                   
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
    printf("File names:\n");

    for(int i = 0; i < (*fileCount); i++)
        printf("\tfile: <%s>\n", *(tempFilenames + i));

    // save file names
    *fileNames = tempFilenames;

    return EXIT_SUCCESS;
}

/**
 *  @brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  @param cmdName string with the name of the command
 */
static void printUsage(char *cmdName) {
    fprintf (stderr,
        "\nSynopsis: %s OPTIONS [filename / positive number]\n"
        "  OPTIONS:\n"
        "  -h      --- print this help\n"
        "  -f      --- filename\n"
        "  -n      --- positive number\n", cmdName);
}