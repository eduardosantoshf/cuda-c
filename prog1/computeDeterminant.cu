#include "computeDeterminant.cuh"

/**
 * @brief Calculates the determinant of a square matrix using Gaussian Elimination
 *
 * This determination transforms the matrix to a base triangular matrix, row by row.
 *
 * @param order order of a determinant
 * @param matrix matrix of 1 Dimension with the length = order * order
 * @return double value of the determinant
 */
double computeDeterminant(int order,  double *matrix) {
    double determinant = 1;
    double pivotElement;
    int pivotRow;

    for (int i = 0; i < order; ++i) {
        pivotElement = matrix[ (i * order) + i]; // current diagonal element
        pivotRow = i;
   
        for (int row = i + 1; row < order; ++row) {
            if (fabs(matrix[(row * order) + i]) > fabs(pivotElement)) {
                // update the value of the pivot and pivot row index
                pivotElement = matrix[(row * order) + i];
                pivotRow = row;
            }
        }
        
        if (pivotElement == 0.0)
            return 0.0; // if diagonal is zero => determinant is zero

        // swap the columns
        if (pivotRow != i) { 
            for (int k = 0; k < order; k++) {
                double temp;
                temp = matrix[(i * order) + k];
                matrix[(i * order) + k] = matrix[(pivotRow * order) + k];
                matrix[(pivotRow * order) + k] = temp;
            }

            determinant *= -1.0; // because of the swap
        }

        determinant *= pivotElement; // update the determinant with the the diagonal value of the current row

        // reduce the matrix to a upper triangle matrix
        for (int row = i + 1; row < order; ++row) {
        // as the current row and column "i" will no longer be used, we may start reducing on the next row/column (i+1)
            for (int col = i + 1; col < order; ++col)
                matrix[(row * order) + col] -= matrix[(row * order) + i] * matrix[(i * order) + col] / pivotElement;
        }
    }

    return determinant;
}