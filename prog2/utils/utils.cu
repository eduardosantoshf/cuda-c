#include "utils.cuh"

/**
 * @brief Reads all the matrixes from a give file.
 *
 * @param fileName name of the file
 * @param matrixArray pointer to array of matrices
 * @param order order of the matrices
 * @param amount total amount of matrices
 */
void readData(char *fileName, double **matrixArray, int *order, int *amount)
{
    FILE *f = fopen(strcat(strdup("../computeDet/"), fileName), "rb");
    if(!f)
    {
        perror("error opening file");
        exit(EXIT_FAILURE);
    }

    if(!fread(amount, sizeof(int), 1, f))
    {
        perror("error reading amount of matrixes");
        exit(EXIT_FAILURE);
    }

    if(!fread(order, sizeof(int), 1, f))
    {
        perror("error reading order of matrixes");
        exit(EXIT_FAILURE);
    }

    (*matrixArray) = (double*)malloc(sizeof(double) * (*amount) * (*order) * (*order));
    if(!(*matrixArray))
    {
        perror("error allocating memory for matrixes");
        exit(EXIT_FAILURE);
    }

    if(!fread((*matrixArray), sizeof(double), (*amount) * (*order) * (*order), f))
    {
        perror("error reading all the matrixes");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Calculates the determinant of a square matrix using Gaussian Elimination
 *
 * This determination transforms the matrix to a base triangular matrix, row by row.
 *
 * @param order the order of a determinant
 * @param matrix the matrix of 1 Dimension with the length of "order" * "order"
 * @return double value of the determinant
 */
double row_by_row_determinant(int order,  double *matrix)
{
    double det = 1;
    double pivotElement;
    int pivotRow;
    for (int i = 0; i < order; ++i)
    {

        pivotElement = matrix[ (i * order) + i]; // current diagonal element
        pivotRow = i;
        // partial pivoting, which should select
        // the entry with largest absolute value from the column of the matrix
        // that is currently being considered as the pivot element
        for (int row = i + 1; row < order; ++row)
        {
            if (fabs(matrix[(row * order) + i]) > fabs(pivotElement))
            {
                // update the value of the pivot and pivot row index
                pivotElement = matrix[(row * order) + i];
                pivotRow = row;
            }
        }
        //if the diagonal element is zero then the determinant will be zeero
        if (pivotElement == 0.0)
        {
            return 0.0;
        }

        if (pivotRow != i) // if the pivotELement is not in the current row, then we perform a swap in the columns
        {
            for (int k = 0; k < order; k++)
            {
                double temp;
                temp = matrix[(i * order) + k];
                matrix[(i * order) + k] = matrix[(pivotRow * order) + k];
                matrix[(pivotRow * order) + k] = temp;
            }

            det *= -1.0; //signal the row swapping
        }

        det *= pivotElement; //update the determinant with the the diagonal value of the current row

        for (int row = i + 1; row < order; ++row) /* reduce the matrix to a  Upper Triangle Matrix */
        // as the current row and column "i" will no longer be used, we may start reducing on the next
        // row/column (i+1)
        {
            for (int col = i + 1; col < order; ++col)
            {
                matrix[(row * order) + col] -= matrix[(row * order) + i] * matrix[(i * order) + col] / pivotElement;  //reduce the value
            }
        }
    }

    return det;
}