#include "computeDeterminant.cuh"

/**
 * @brief Calculates the determinant of a square matrix using Gaussian Elimination
 *
 * This determination transforms the matrix to a base triangular matrix, column by column.
 *
 * @param order order of a determinant
 * @param matrix matrix of 1 Dimension with the length = order * order
 * @return double value of the determinant
 */
double computeDeterminant(int order,  double *matrix) {
    double determinant = 1;
    double pivotElement;
    int pivotColumn;
    for (int i = 0; i < order; ++i){
        pivotElement = matrix[ (i * order) + i]; // current diagonal element
        pivotColumn = i;
        // partial pivoting, which should select
        // the entry with largest absolute value from the column of the matrix
        // that is currently being considered as the pivot element
        for (int column = i + 1; column < order; ++column) {
            if (fabs(matrix[(i * order) + column]) > fabs(pivotElement)) {
                // update the value of the pivot and pivot column index
                pivotElement = matrix[(i * order) + column];
                pivotColumn = column;
            }
        }

        //if the diagonal element is zero then the determinant will be zeero
        if (pivotElement == 0.0)
            return 0.0;

        if (pivotColumn != i) { // if the pivotELement is not in the current column, then we perform a swap in the rows
            for (int k = 0; k < order; k++) {
                double temp;
                temp = matrix[(k * order) + i];
                matrix[(k * order) + i] = matrix[(k * order) + pivotColumn];
                matrix[(k * order) + pivotColumn] = temp;
            }

            determinant *= -1.0; //signal the row swapping
        }

        determinant *= pivotElement; //update the determinant with the the diagonal value of the current row

        for (int column = i + 1; column < order; ++column) { /* reduce the matrix to a  Base Triangle Matrix */
        // as the current row and column "i" will no longer be used, we may start reducing on the next
        // column/row (i+1)
            for (int row = i + 1; row < order; ++row)
                matrix[(row * order) + column] -= matrix[(row * order) + i] * matrix[(i * order) + column] / pivotElement;  //reduce the value
        }
    }

    return determinant;
}