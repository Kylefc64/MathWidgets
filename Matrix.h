//#pragma once
#ifndef MATRIX_H
#define MATRIX_H
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include "ComplexNumber.h"
#include <omp.h>
#include <intrin.h>

class Matrix {
public:
	Matrix(const std::string&); //must be parsed
	Matrix(const Matrix&); //copy constructor
	Matrix(Matrix&&); //move constructor
	Matrix(const int&, const int&); //creates a matrix of zeros
	Matrix(ComplexNumber**, const int&, const int&); //mostly to be used for testing purposes
	~Matrix();
	Matrix& operator=(const Matrix&); //copy assignment operator
	Matrix& operator=(Matrix&&); //move assignment operator
	ComplexNumber& operator()(const int&, const int&); //access/mutation operator
	friend const Matrix operator+(const Matrix&, const Matrix&); //element-wise addition
	friend const Matrix operator-(const Matrix&, const Matrix&); //element-wise subtraction
	friend const Matrix operator*(const Matrix&, const Matrix&); //element-wise multiplication
	friend const Matrix operator/(const Matrix&, const Matrix&); //element-wise division
	const Matrix& operator+=(const Matrix&);
	const Matrix& operator-=(const Matrix&);
	const Matrix& operator*=(const Matrix&);
	const Matrix& operator/=(const Matrix&);
	const Matrix dot(const Matrix&) const; //matrix multiplication
	const Matrix inverse() const; //calculates the inverse of this matrix, if it exists
	Matrix augment(const Matrix&) const; //augments m to the right of this matrix
	const Matrix identity() const; //returns the identity matrix of the same size as this matrix, if it exists
	const static Matrix identity(const int&); //returns an identity matrix of specified size
	const Matrix transpose() const; //calculates the transpose of this matrix, if it exists
	const Matrix conjugateTranspose() const; //calculates the conjugate transpose of a matrix
	//throws an Exception if dimensions of matrices do not match
	static void checkDimensions(const Matrix&, const Matrix&);
	static Matrix solve(const Matrix&, const Matrix&); //calculates the solution to Ax=b
	static Matrix leastSquares(const Matrix&, const Matrix&); //calculates x = ((aTa)^-1)aTb
	friend std::ostream& operator<<(std::ostream&, const Matrix&); //Inserts the contents of a Matrix into an ostream
	friend std::istream& operator >> (std::istream&, Matrix&); //Extracts Matrix data from an istream and inserts it into a Matrix object
	Matrix rowReduce() const; //returns a row-reduced matrix
	Matrix verticalPad(const int&) const; //returns a Matrix zero-padded on the bottom by the specified size
	Matrix horizontalPad(const int&) const; //returns a Matrix zero-padded on the right by the specified size
	Matrix nullSpace() const;

	//Matrix LSCSort() const; //least significant column sort
private:
	int rows;
	int cols;
	ComplexNumber** matrix;
	//std::vector<std::string> parseRow(const std::string&); //returns a vector containing elements of a matrix row
	int findNextNonZeroEntry(const int&); //returns the row of the first non-zero entry in the specified column, or -1
	bool allZeros(ComplexNumber*, const int&, const int&) const; //returns true if the entire row (from start to end) is 0
	void swap(ComplexNumber*&, ComplexNumber*&) const; //swaps the contents of each row
	int findNextZeroRow(const int&) const; //finds the next row full of zeros starting at the specified index
};

#endif