#include "SquareMatrix.h"
#include "Matrix.h"
SquareMatrix::SquareMatrix(const std::string& s) : Matrix(s) {
	if (rows != cols) {
		// TODO: throw exception (not a square matrix)
	}
}
SquareMatrix::SquareMatrix(ComplexNumber** m, const size_t& size) : Matrix(m, size, size) {
	if (rows != cols) {
		// TODO: throw exception (not a square matrix)
	}
}
/* Returns a new SquareMatrix that is the inverse of this SquareMatrix. */
const SquareMatrix SquareMatrix::inverse() const {
	Matrix augmented = augment(identity()).rowReduce();
	/*Matrix augmented = augment(identity());
	for (size_t j = 0; j < cols; j++) {
	size_t nextRow = augmented.findNextNonZeroEntry(j);
	if (nextRow == -1) { //the entire column has 0 entries
	continue;
	}
	else if (nextRow != j) { //keeps the pivots in the correct rows/columns
	swap(augmented.matrix[nextRow], augmented.matrix[j]);
	nextRow = j;
	}
	//divide row nextRow by the first non-zero entry in that row
	//this results in a "1" in column j
	ComplexNumber divisor = augmented.matrix[nextRow][j];
	for (size_t k = j; k < 2 * cols; k++) {
	augmented.matrix[nextRow][k] /= divisor;
	}
	//zero out column j of every row except for nextRow
	for (size_t i = 0; i < rows; i++) {
	if (i != nextRow) {
	ComplexNumber multiple = augmented.matrix[i][j];
	for (size_t k = j; k < 2 * cols; k++) {
	augmented.matrix[i][k] -= multiple * augmented.matrix[nextRow][k];
	}
	}
	}
	}*/
	//verify that the matrix is indeed invertible (that all rows are non-zero after Gaussian-elimination)
	for (size_t i = 0; i < rows; i++) {
		if (augmented.allZeros(i, 0, 2 * cols)) {
			//TODO: throw exception
		}
	}
	//de-augment the matrix
	ComplexNumber** c = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (size_t i = 0; i < rows; i++) {
		c[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			c[i][j] = augmented(i, j + cols);
		}
	}
	return SquareMatrix(c, rows);
}
/* Returns a new identity SquareMatrix that has the same dimensions of this matrix. */
const SquareMatrix SquareMatrix::identity() const {
	return identity(rows);
}
/* Returns a new identity SquareMatrix of the given dimensions. */
const SquareMatrix SquareMatrix::identity(const size_t& size) {
	ComplexNumber** c = new ComplexNumber*[size];
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		c[i] = new ComplexNumber[size];
		for (size_t j = 0; j < size; j++) {
			if (i == j) {
				c[i][j] = 1;
			}
		}
	}
	return SquareMatrix(c, size);
}