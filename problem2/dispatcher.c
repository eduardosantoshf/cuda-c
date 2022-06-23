#include "dispatcher.h"

void storePartialResult(double **results, int fileId, int matrixId, double determinant) {
    results[fileId][matrixId] = determinant; /* store value */
}
