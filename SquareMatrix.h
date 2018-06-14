#ifndef SQUARE_MATRIX_H
#define SQUARE_MATRIX_H
#include "Matrix.h"
#include <string>
/* Represents a square matrix. Once a SquareMatrix is constructed, all operations
	on a SquareMatrix can safely assume that the matrix is indeed square. */
class SquareMatrix : public Matrix {
public:
	SquareMatrix(const std::string&); //constructs a SquareMatrix by parsing an input string
	SquareMatrix(ComplexNumber**, const int&); //mostly for testing purposes
	const SquareMatrix inverse() const; //calculates the inverse of this matrix, if it exists
	const SquareMatrix identity() const; //returns the identity matrix of the same size as this matrix, if it exists
	const static SquareMatrix identity(const int&); //returns an identity matrix of specified size
};
#endif
