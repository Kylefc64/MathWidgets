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
#include "Contiguous2DArray.h"

class Matrix {
public:
	Matrix(const std::string&); //must be parsed
	Matrix(const Matrix&); //copy constructor
	Matrix(Matrix&&); //move constructor
	Matrix(const size_t&, const size_t&); //creates a matrix of zeros
	Matrix(const Contiguous2DArray<ComplexNumber>&, const size_t&, const size_t&); //mostly to be used for testing purposes
	~Matrix();
	Matrix& operator=(const Matrix&); //copy assignment operator
	Matrix& operator=(Matrix&&); //move assignment operator
	ComplexNumber& operator()(const size_t&, const size_t&); //access/mutation operator
	friend const Matrix operator+(const Matrix&, const Matrix&); //element-wise addition
	friend const Matrix operator-(const Matrix&, const Matrix&); //element-wise subtraction
	friend const Matrix operator*(const Matrix&, const Matrix&); //element-wise multiplication
	friend const Matrix operator/(const Matrix&, const Matrix&); //element-wise division
	const Matrix& operator+=(const Matrix&);
	const Matrix& operator-=(const Matrix&);
	const Matrix& operator*=(const Matrix&);
	const Matrix& operator/=(const Matrix&);
	Matrix dot(const Matrix&) const; //matrix multiplication
	Matrix augment(const Matrix&) const; //augments m to the right of this matrix
	const Matrix transpose() const; //calculates the transpose of this matrix, if it exists
	const Matrix conjugateTranspose() const; //calculates the conjugate transpose of a matrix
	static void checkDimensions(const Matrix&, const Matrix&); //throws an exception if dimensions of matrices do not match
	static Matrix leastSquares(const Matrix&, const Matrix&); //calculates x = ((aTa)^-1)aTb
	friend std::ostream& operator<<(std::ostream&, const Matrix&); //Inserts the contents of a Matrix into an ostream
	friend std::istream& operator >> (std::istream&, Matrix&); //Extracts Matrix data from an istream and inserts it into a Matrix object
	Matrix rowReduce() const; //returns a row-reduced matrix
	Matrix verticalPad(const size_t&) const; //returns a Matrix zero-padded on the bottom by the specified size
	Matrix horizontalPad(const size_t&) const; //returns a Matrix zero-padded on the right by the specified size
	Matrix nullSpace() const;
	bool allZeros(const size_t&, const size_t&, const size_t&) const; //returns true if the entire row (from start to end) is 0
	size_t numRows() const;
	size_t numCols() const;

	//Matrix LSCSort() const; //least significant column sort

protected:
	Contiguous2DArray<ComplexNumber> matrix_;
	ComplexNumber* getRow(const size_t&) const; // returns the ith row of matrix (assumes that i is a valid row)
	size_t findNextNonZeroEntry(const size_t&); // returns the row of the first non-zero entry in the specified column, or -1
	void copyMatrix(Contiguous2DArray<ComplexNumber>&, const Contiguous2DArray<ComplexNumber>&, const size_t&, const size_t&) const;

private:
	//std::vector<std::string> parseRow(const std::string&); // returns a vector containing elements of a matrix row
	void swap(ComplexNumber*, ComplexNumber*, const size_t&) const; // swaps the contents of each row
	size_t findNextZeroRow(const size_t&) const; // finds the next row full of zeros starting at the specified index
	const std::vector<std::vector<std::string>> parseRowVectors(const std::string) const;
	void initMatrix(const std::vector<std::vector<std::string>>&);
};
#endif
