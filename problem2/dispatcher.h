
#ifndef DISPATCHER_H
#define DISPATCHER_H
#include <stdio.h>

/**
 *  \file dispatcher.h
 *
 *  @brief Struct to be used for each matrix.
 *
 *  \author Eduardo Santos and Pedro Bastos - May 2022
 * 
 */
typedef struct chunkInfo{
    int fileId;
    int matrixId;
    double* matrixPtr; 
    int order;
    int isLastChunk;
} chunkInfo, *pChunkInfo;

/**
 * \file dispatcher.h
 * 
 * @brief Method to save partial results from each matrix
 * 
 * @param fileId the Identifier of the File where the results will be put
 * @param matrixId  the Identifier of the Matrix for which the Determinant was calculated
 * @param determinant the determinant value obtained to be stored
 */
void storePartialResult(double **results, int fileId, int matrixId, double determinant);

/**
 *  \file dispatcher.h
 *
 *  @brief Method to print the results
 *
 * @param results Determinants
 * @param matrixId Number of matrices in file
 */
void printResults(double **results,int fileAmount);

#endif