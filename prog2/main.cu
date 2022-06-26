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
void readData(char *fileName, double **matrixArray, int *order, int *amount);

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
    int fileAmount = 0;
    char ** fileNames;
    int commandResult;

    // process the command and act according to it
    commandResult = processCommand(argc, argv, &fileAmount, &fileNames);

    if (commandResult != EXIT_SUCCESS) {
        perror("There was an error processing the input");
        exit(EXIT_FAILURE);
    }

    // device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // process files
    double *hostMatrix = NULL;
    int order = 0, amount = 0;
    
    for (int i = 0; i < fileAmount; i++) {
        // read data from file
        readData(*(fileNames + i), &hostMatrix, &order, &amount);

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
        dim3 grid(amount, 1, 1);
        dim3 block(order, 1, 1);

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
 * @brief Computes the determinant column by column on the host
 *
 * @param matrix pointer to matrix
 * @param order order of matrix
 */
void computeDeterminantHost(int order, int amount, double **matrix, double *results) {
    for (int i = 0; i < amount; i++)
        *(results + i) = computeDeterminant(order, ((*matrix) + (i * order * order)));
}

/**
 * @brief Computes the determinant column by column on the GPU
 *
 * @param deviceMatrix matrices array
 * @param deviceResults results array
 */
__global__ void computeDeterminantGPU(double *deviceMatrix, double *deviceResults) {
    int order = blockDim.x;
    int matrixIdx = blockIdx.x * order * order;
    int tColumn = threadIdx.x + matrixIdx;
    int pivotColumn;

    for (int iteration = 0; iteration < order; iteration++) {
        if (threadIdx.x < iteration)
            return;

        int iterColumn = iteration + matrixIdx;
        double pivot = deviceMatrix[iterColumn + iteration * order];
        pivotColumn = iterColumn;

        if (threadIdx.x == iteration) {
            for (int col = iterColumn + 1; col < ( matrixIdx + order); ++col) {
                if (fabs(deviceMatrix[(iteration * order) + col]) > fabs(pivot)) {
                    // update the value of the pivot and pivot col index
                    pivot = deviceMatrix[(iteration * order) + col];
                    pivotColumn = col;

	      	    }
            }

            if (iteration == 0)
                deviceResults[blockIdx.x] = 1;

            if (pivotColumn != iterColumn) {
                for (int k = 0; k < order; k++) {
                    double temp;
                    temp = deviceMatrix[(k * order) + iterColumn];
                    deviceMatrix[(k * order) + iterColumn] = deviceMatrix[(k * order) + pivotColumn];
                    deviceMatrix[(k * order) + pivotColumn] = temp;
                }

                deviceResults[blockIdx.x] *= -1.0; // signal the row swapping
            }
                deviceResults[blockIdx.x] *= pivot;

            return;
        }
        __syncthreads();
	    
        iterColumn = iteration + matrixIdx;
        pivot = deviceMatrix[iterColumn + iteration * order];
        pivotColumn = iterColumn;

        double const_val = deviceMatrix[tColumn + order * iteration] / pivot;

        for (int row = iteration + 1; row < order; row++)
            deviceMatrix[tColumn + order * row] -= deviceMatrix[pivotColumn + order * row] * const_val;

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
void readData(char *fileName, double **matrixArray, int *order, int *amount) {
    FILE *f = fopen(fileName, "rb");
    
    if (!f) {
        perror("error opening file");
        exit(EXIT_FAILURE);
    }

    if (!fread(amount, sizeof(int), 1, f)) {
        perror("error reading amount of matrixes");
        exit(EXIT_FAILURE);
    }

    if (!fread(order, sizeof(int), 1, f)) {
        perror("error reading order of matrixes");
        exit(EXIT_FAILURE);
    }

    (*matrixArray) = (double*)malloc(sizeof(double) * (*amount) * (*order) * (*order));
    
    if (!(*matrixArray)) {
        perror("error allocating memory for matrixes");
        exit(EXIT_FAILURE);
    }

    if (!fread((*matrixArray), sizeof(double), (*amount) * (*order) * (*order), f)) {
        perror("error reading all the matrixes");
        exit(EXIT_FAILURE);
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
int processCommand(int argc, char *argv[], int* fileAmount, char*** fileNames) {
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
static void printUsage(char *cmdName) {
    fprintf (stderr,
        "\nSynopsis: %s OPTIONS [filename / positive number]\n"
        "  OPTIONS:\n"
        "  -h      --- print this help\n"
        "  -f      --- filename\n"
        "  -n      --- positive number\n", cmdName);
}