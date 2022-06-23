#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>
#include <sys/types.h>
#include <stdbool.h>
#include <math.h>
#include "worker.h"
#include "dispatcher.h"

/* General definitions */
#define WORKTODO 1
#define NOMOREWORK 0

/** \brief number of workers */
int nWorkers;
/** \brief  Array containing the amount of matrices **/
int *matrixAmount;
/** \brief dispatcher life cycle routine */
void dispatcher(char ***fileNames, int fileAmount);
/** \brief worker life cycle routine */
void work(int rank);

/** \brief process the called command. */
static int process_command(int argc, char *argv[], int* , char*** fileNames);

/** \brief Print the explanation of how to use the command. */
static void printUsage (char *cmdName);

/**
 * \brief Main method
 *
 * Will read and validate the input.
 * Its role is to start the MPI, which will process the matrices
 */
int main(int argc, char *argv[]) {
    struct timespec start, finish; // time limits

    int fileAmount = 0;            // amount of files
    char **fileNames;             // array of pointers, each pointer points to a string literal

    int rank, // process rank
        size; // amout of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nWorkers = size - 1;

    if (rank == 0) { // root process (dispatcher)
        clock_gettime(CLOCK_MONOTONIC_RAW, &start); // start counting time
    
        // will hold the output of processing the command
        int command_result;

        // process the command and act according to it
        command_result = process_command(argc, argv, &fileAmount, &fileNames);
        if (command_result != EXIT_SUCCESS) {
            free(fileNames);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // kill every living process (root included)
        } else {
            dispatcher(&fileNames, fileAmount);
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish); // end counting time

        // calculate execution time
        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nRoot elapsed time = %.6f s\n", executionTime); // print execution time
    }
    else {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start); // start counting time

        work(rank); // worker logic

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish); // end counting time

        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nWorker <%d> elapsed time = %.6f s\n", rank, executionTime); // print execution time
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

/**
 * \brief Dispatcher
 *
 * Will read and process the files, sending work to th workers, storing and printing the results
 * 
 * @param fileNames Files
 * @param fileAmount Number of files to be processed
 */
void dispatcher(char ***fileNames, int fileAmount) {
    int workerId;
    unsigned int ToDo = WORKTODO;
    double **results = malloc(fileAmount * sizeof(double *));
    matrixAmount = malloc(fileAmount * sizeof(int));
    char **file_names = (*fileNames);
    int amount = 0, order = 0, lastChunkSize = 0;
    int matrixId = 0, fileId = 0, chunkId = 0;
    int chunksToSend;

    FILE *f = NULL;

    while (ToDo) {
        /* If there are no more files to process*/
        if (fileId == fileAmount) {
            printf("No more work, sending message to workers to end..\n");
            ToDo = NOMOREWORK;

            for (int i = 1; i <= nWorkers; i++)
                MPI_Send(&ToDo, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD); /* tell workers there is no more work to be done */
            break;
        }
        /* Otherwise tell workers that there is work to do..*/
        for (int i = 1; i <= nWorkers; i++)
            MPI_Send(&ToDo, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        
        /* If the pointer is null open the File*/
        if (f == NULL) {
            f = fopen(file_names[fileId], "r");

            if (f == NULL) {
                printf("Could not open file\n");
                exit(-1);
            }

            amount = 0;

            // reads amount of matrices
            if (!fread(&amount, sizeof(int), 1, f)) {
                printf("Error reading amount. Exiting...\n");
                exit(-1);
            }

            results[fileId] = malloc(amount * sizeof(double));
            matrixAmount[fileId] = amount;
            // reads order of matrices
            order = 0;

            if (!fread(&order, sizeof(int), 1, f)) {
                printf("Error reading order. Exiting...");
                exit(-1);
            }

            lastChunkSize = matrixAmount[fileId] % (nWorkers);
        }
        chunksToSend = chunkId == (matrixAmount[fileId] - lastChunkSize) ? lastChunkSize : (nWorkers);

        for (int j = 1; j <= nWorkers; j++) {
            chunkInfo chunk;
            if (matrixAmount[fileId] == chunkId) {
                chunk.isLastChunk=1;
                MPI_Send(&chunk, sizeof(chunkInfo), MPI_BYTE, j, 0, MPI_COMM_WORLD);
                continue;
            }
            else {
                chunk.isLastChunk = 0;
                chunk.fileId = fileId;
                chunk.order = order;
                chunk.matrixId = matrixId;
                matrixId++;

            }
            MPI_Send(&chunk, sizeof(chunkInfo), MPI_BYTE, j, 0, MPI_COMM_WORLD);
            double *matrix = (double *)malloc(sizeof(double) * order * order);
            
            if (!fread(matrix, 8, order * order, f))
                break;

            //if(x){
            //    x = false;
            //    for(int i = 0; i < order*order; i++){
            //        printf("%f ", matrix[i]);
            //    }
            //}

            MPI_Send(matrix, order * order, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
            chunkId++;

            free(matrix);
        }

        for (workerId = 1; workerId <= chunksToSend; workerId++) {
            double partialResultData[3]; /* received partial info computed by workers */
            MPI_Recv(partialResultData, 3, MPI_DOUBLE, workerId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            storePartialResult(results, partialResultData[1], partialResultData[2], partialResultData[0]);
        }
        if (chunkId == matrixAmount[fileId]) {
            fclose(f);
            f = NULL;
            fileId++;
        }
    }
    printResults(results, fileAmount);
} 

/**
 *
 * This method will compute the determinat a store itt as partial info, sending it to dispatcher
 * 
 * @param rank process rank
 */
void work(int rank) {
    unsigned int ToDo; /* command */
    double determinant_result;
    int order;
    double *matrix;
    chunkInfo chunk;

    while (true)
    {
        MPI_Recv(&ToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ToDo == NOMOREWORK) { // no more work to be done by workers
        
            printf("Worker with rank %d terminated...\n", rank);
            return;
        }

        MPI_Recv(&chunk, sizeof(chunkInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (chunk.isLastChunk){
            printf("Last Chunk..\n");
            continue;
        }

        order = chunk.order;
        matrix = (double *)malloc(sizeof(double) * order * order);
        MPI_Recv(matrix, order * order, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        determinant_result = computeDeterminant(order, matrix);

        double partialResultData[3];
        partialResultData[0] = determinant_result;
        partialResultData[1] = chunk.fileId;
        partialResultData[2] = chunk.matrixId;

        MPI_Send(partialResultData, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // send partial info computed to dispatcher

        free(matrix);
    }
}

/**
 * @brief Method invoked by Dispatcher to Print the final Results gathered from all workers
 *
 * @param fileAmount the amount of files used to create and process chunks
 * @param matrixAmount the amount of matrices
 */
void printResults(double **results, int fileAmount) {
    for (int i = 0; i < fileAmount; i++) {
        printf("File nº: <%d>\n", i + 1);
        for (int j = 0; j < matrixAmount[i]; j++)
            printf("The determinant for matrix nº %d is %+5.3e \t\n", j + 1, results[i][j]);
    }
    //printf("The determinant for matrix nº 1 is %+5.3e \t\n", results[0][0]);

    free(results);
    free(matrixAmount);
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
int process_command(int argc, char *argv[], int* fileAmount, char*** fileNames) {
    char **auxFileNames = NULL; 
    int opt; // selected option

    if(argc <= 2) {
        perror("No/few arguments were provided.");
        printUsage(basename("PROGRAM"));
        return EXIT_FAILURE;
    }

    opterr = 0;
    do { 
        switch ((opt = getopt (argc, argv, "f:h"))) { 
            case 'f':                                                   // case: file name
                if (optarg[0] == '-') { 
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename (argv[0]));
                    return EXIT_FAILURE;
                }

                int index = optind - 1;
                char* next = NULL;

                while(index < argc) {
                    next = argv[index++];                               // get next element in argv

                    if(next[0] != '-') {                               // if element isn't an option, then its a file name
                        if((*fileAmount) == 0) {                          // first file name
                            auxFileNames = malloc(sizeof(char*) * (++(*fileAmount)));
                            
                            if(!auxFileNames) {                           // error reallocating memory
                                fprintf(stderr, "error allocating memory for file name\n");
                                return EXIT_FAILURE;
                            }

                            *(auxFileNames + (*fileAmount) - 1) = next;
                        }
                        else {                                            // following file names
                            (*fileAmount)++;
                            auxFileNames = realloc(auxFileNames, sizeof(char*) * (*fileAmount));

                            if(!auxFileNames) {                           // error reallocating memory
                                fprintf(stderr, "error reallocating memory for file name\n");
                                return EXIT_FAILURE;
                            }

                            *(auxFileNames + (*fileAmount) -1) = next;
                        }
                    }
                    else
                        break;
                }
                break;

            case 'h':                                                   // case: help mode
                printUsage (basename (argv[0]));
                return EXIT_SUCCESS;

            case '?':                                                   // case: invalid option
                fprintf(stderr, "%s: invalid option\n", basename (argv[0]));
                printUsage(basename (argv[0]));
                return EXIT_FAILURE;

            default:  
                break;
        }

    } while (opt != -1);

    printf("File amount: <%d>\nFile names:\n", (*fileAmount));

    for(int i = 0; i < (*fileAmount); i++) {
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