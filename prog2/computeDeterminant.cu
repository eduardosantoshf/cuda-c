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
    
    double pivotElement;
    double determinant = 1;
    int pivotColumn;
    int i, column, row, k, new_index;
    bool swap = false;
    double scale_value;

    for (i = 0; i < order; ++i){

        pivotElement = matrix[(i * order) + i]; 
        pivotColumn = i;

        // select the largest pivot of the current pivot column
        for (column = i + 1; column < order; ++column) {
            new_index = (i * order) + column;
            
            // if greater than current value, update pivot
            if (fabs(matrix[new_index]) > fabs(pivotElement)) {
                pivotElement = matrix[new_index];
                pivotColumn = column;
            }
        }

        // det = 0
        if (pivotElement == 0.0)
            return 0.0;

        // if another column was selected as pivot
        if (pivotColumn != i) { 
            swap = true;

            // swap 
            for (k = 0; k < order; k++) {
                double temp;
                temp = matrix[(k * order) + i];
                matrix[(k * order) + i] = matrix[(k * order) + pivotColumn];
                matrix[(k * order) + pivotColumn] = temp;
            }
            
        }

        //if there was a swap
        if(swap){
            swap = false;
            determinant *= -1.0; 
        }

        // update final determinant
        determinant *= pivotElement; 

        // reduce to a base triangle matrix
        for (column = i + 1; column < order; ++column) {
            
            // scale to multiply the elements
            scale_value = matrix[(i * order) + column] / pivotElement;

            // reduce the matrix values (ignoring the previously used column and row)
            for (row = i + 1; row < order; ++row)
                matrix[(row * order) + column] -= scale_value * matrix[(row * order) + i];  
        }
    }

    return determinant;
}