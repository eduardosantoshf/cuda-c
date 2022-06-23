#include <math.h>
#include <stdio.h>
#include "worker.h"

#include <stdlib.h>

double computeDeterminant(int order,  double * matrix) {
    double det = 1;
    //double pivotElement;
    //int pivotRow;

    double **matrix_bi;

    matrix_bi = malloc(order * sizeof(double *));
    for(int z = 0; z < order; z++){
        matrix_bi[z] = malloc(order * sizeof(double *));
    }
    
    int i, j, k;
    double ratio;

    printf("ola");


    for(int x = 0; x < order; x++){ 
        for(int y = 0; y < order; y++){
            matrix_bi[x][y] = matrix[(x * order) + y];
        }
    }
    

    // primeira maneira
    //for (i = 0; i < order; i++)
    //{
    //    if (matrix_bi[i][i] == 0.0)
    //    {
    //        printf("Mathematical Error!");
    //        return 0;
    //        //exit(0);
    //    }
    //    for (j = i + 1; j < order; j++)
    //    {
    //        ratio = matrix_bi[j][i] / matrix_bi[i][i];
//
    //        for (k = 0; k < order; k++)
    //        {
    //                matrix_bi[j][k] = matrix_bi[j][k] - ratio * matrix_bi[i][k];
    //        }
    //    }
    //}

    //segunda maneira
    for (i = 0; i < order; i++)
    {
        if (matrix_bi[i][i] == 0.0)
        {
            printf("Mathematical Error!");
            return 0;
            //exit(0);
        }
        for (j = i + 1; j < order; j++)
        {
            ratio = matrix_bi[i][j] / matrix_bi[i][i];

            for (k = 0; k < order; k++)
            {
                    matrix_bi[k][j] = matrix_bi[k][j] - ratio * matrix_bi[k][i];
            }
        }
    }


    // Finding determinant by multiplying
    // elements in principal diagonal elements 
    for (i = 0; i < order; i++)
        det = det * matrix_bi[i][i];

    return det;
}