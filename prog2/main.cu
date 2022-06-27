#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <libgen.h>

#include "common.h"
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
static int processCommand(int argc, char *argv[], int* , char*** file_names);

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
    int file_count = 0;
    char ** file_names;
    int commandResult;

    // process the command and act according to it
    commandResult = processCommand(argc, argv, &file_count, &file_names);

    if (commandResult != EXIT_SUCCESS) {
        perror("There was an error processing the command line");
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
    int i;
    
    for (i = 0; i < file_count; i++) {

        // read files
        fileHandling(*(file_names + i), &hostMatrix, &order, &amount);

        double *retrievedResults = (double *)malloc(sizeof(double) * amount);
        double *deviceMatrix;
        double *deviceResults;
        
        // allocate memory
        CHECK(cudaMalloc((void **)&deviceMatrix, (sizeof(double) * order * order * amount)));
        CHECK(cudaMalloc((void **)&deviceResults, sizeof(double) * amount));

        // copy the data
        CHECK(cudaMemcpy(deviceMatrix, hostMatrix, (sizeof(double) * order * order * amount), cudaMemcpyHostToDevice));

        // create grid and block
        dim3 grid(amount); // by omission, grid(amount) <=> grid(amount, 1, 1)
        dim3 block(order); // // by omission, block(order) <=> block(order, 1, 1)

        
        double deviceStart = seconds();

        // process on device
        computeDeterminantGPU<<<grid, block>>>(deviceMatrix, deviceResults);
        CHECK(cudaDeviceSynchronize());

        double deviceTime = seconds() - deviceStart;

        CHECK(cudaGetLastError()); 
        // return results
        CHECK(cudaMemcpy(retrievedResults, deviceResults, sizeof(double) * amount, cudaMemcpyDeviceToHost)); 
        // free memory 
        CHECK(cudaFree (deviceMatrix)); 

        // process on host
        double hostResults[amount];
        double hostStart = seconds();

        computeDeterminantHost(order, amount, &hostMatrix, hostResults);

        double hostTime = seconds() - hostStart;

        printf("\nRResults:\n");

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
    int pivotCol;
    int iter, col, k, row, new_index;
    double temp_value, scale_val;
    bool swap = false;
    int order = blockDim.x;
    int matrixIdx = blockIdx.x * order * order;
    int iterCol;
    double pivot;
    
    

    for (iter = 0; iter < order; iter++) {

        iterCol = iter + matrixIdx; // current iteration

        pivotCol = iterCol;
        pivot = deviceMatrix[iterCol + iter * order];
        
        // each thread only deals with one partial pivot
        if (threadIdx.x == iter) {
            
            // initialize 
            if (iter == 0)
                deviceResults[blockIdx.x] = 1;

            // select the largest pivot of the current pivot column
            for (col = iterCol + 1; col < (matrixIdx + order); ++col) {
                new_index = (iter * order) + col;

                // if greater than current value, update pivot
                if (fabs(deviceMatrix[new_index]) > fabs(pivot)) {
                    pivot = deviceMatrix[new_index];
                    pivotCol = col;

	      	    }
            }

            // if another pivot was selected
            if (pivotCol != iterCol) {
                swap = true;

                // swap columns
                for (k = 0; k < order; k++) {
                    temp_value = deviceMatrix[(k * order) + iterCol];
                    deviceMatrix[(k * order) + iterCol] = deviceMatrix[(k * order) + pivotCol];
                    deviceMatrix[(k * order) + pivotCol] = temp_value;   
                }
            }

            // if there was a swap
            if (swap){
                deviceResults[blockIdx.x] *= -1.0; 
                swap = false;
            }
            
            // update final determinant
            deviceResults[blockIdx.x] *= pivot;

            return;
        }
        __syncthreads();
	    
        iterCol = iter + matrixIdx;
        pivot = deviceMatrix[iterCol + iter * order];
        pivotCol = iterCol;

        scale_val = deviceMatrix[(threadIdx.x + matrixIdx) + order * iter] / pivot;

        // reduce to a base triangle matrix
        for (row = iter + 1; row < order; row++)
            deviceMatrix[(threadIdx.x + matrixIdx) + order * row] -= scale_val * deviceMatrix[pivotCol + order * row];

        __syncthreads();

        // not the right iteration for this thread
        if (threadIdx.x < iter)
            return;
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
        perror("ERROR: can't open file.");
        exit(EXIT_FAILURE);
    }

    fread(amount, sizeof(int), 1, f);

    fread(order, sizeof(int), 1, f);

    (*matrixArray) = (double*)malloc(sizeof(double) * (*amount) * (*order) * (*order));
    
    fread((*matrixArray), sizeof(double), (*amount) * (*order) * (*order), f);
}

/**
 * @brief Process command line input
 *
 * Iterates through argv to find and store thread amount, file amount and file names.
 *
 * @param argc Argument quantity in the command line
 * @param argv Array with arguments fromt he command line
 * @param file_count Pointer to file amount
 * @param file_names Pointer to pointer array where file names are stored
 * @return int Return value of command line processing
 */
int processCommand(int argc, char *argv[], int* file_count, char*** file_names) {
    char **temp_filenames = NULL;
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
            {                                                
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
                    // next arg
                    next = argv[index++];                               

                    if((*file_count) == 0)                         
                    {
                        temp_filenames = (char **)malloc(sizeof(char*) * (++(*file_count)));
                        *(temp_filenames + (*file_count) - 1) = next;
                    }
                    else                                            
                    {
                        (*file_count)++;
                        temp_filenames = (char **)realloc(temp_filenames, sizeof(char*) * (*file_count));
                        *(temp_filenames + (*file_count) -1) = next;
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

    for(int i = 0; i < (*file_count); i++) {
        printf("\t%s\n", *(temp_filenames + i));
    }

    // save file names
    *file_names = temp_filenames;

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