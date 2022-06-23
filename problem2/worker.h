#ifndef WORKER_H
#define WORKER_H
#include <stdio.h>

/**
 * \file worker.h
 *
 * @brief Method to compute the determinant of a given matrix
 *
 * \author Eduardo Santos and Pedro Bastos - May 2022
 * 
 * @param order Matrix order
 * @param matrix the matrix of 1 Dimension with the length of "order" * "order"
 * @return double the determinant value
 */
double computeDeterminant(int order, double *matrix);
#endif