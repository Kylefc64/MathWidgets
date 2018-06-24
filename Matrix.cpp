#include "Matrix.h"
#include "SquareMatrix.h"

Matrix::Matrix(const std::string& s) : matrix_(0, 0) {
	//Assumes str is syntactically correct (for now)
	const std::vector<std::vector<std::string>> rowVectors = parseRowVectors(s);
	//initialize the Matrix
	initMatrix(rowVectors);
}
Matrix::Matrix(const size_t& r, const size_t& c) : matrix_(r, c) {}
Matrix::Matrix(const Contiguous2DArray<ComplexNumber>& m, const size_t& r, const size_t& c) :matrix_(r, c) {
	copyMatrix(matrix_, m, r, c);
}
Matrix::Matrix(const Matrix& m) : matrix_(m.numRows(), m.numCols()) {
	copyMatrix(matrix_, m.matrix_, m.numRows(), m.numCols());
}
Matrix::Matrix(Matrix&& m) : matrix_(std::move(m.matrix_)) {
}
Matrix::~Matrix() {}
Matrix& Matrix::operator=(const Matrix& m) {
	if (this == &m) {
		return *this;
	}
	if (numRows() == m.numRows() && numCols() == m.numCols()) {
		//do not reallocate matrix
		copyMatrix(matrix_, m.matrix_, numRows(), numCols());
		return *this;
	}
	matrix_ = Contiguous2DArray<ComplexNumber>(m.numRows(), m.numCols());
	copyMatrix(matrix_, m.matrix_, numRows(), numCols());
	return *this;
}
Matrix& Matrix::operator=(Matrix&& m) {
	//std::cout << "move assignment operator";
	matrix_ = std::move(m.matrix_);
	return *this;
}
ComplexNumber& Matrix::operator()(const size_t& row, const size_t& col) {
	if (row >= numRows() || col >= numCols()) {
		//TODO: throw exception
	}
	return matrix_[row][col];
}
const Matrix operator+(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			m.matrix_[i][j] += m2.matrix_[i][j];
		}
	}
	return m;
}
const Matrix operator-(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			m.matrix_[i][j] -= m2.matrix_[i][j];
		}
	}
	return m;
}
const Matrix operator*(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			m.matrix_[i][j] *= m2.matrix_[i][j];
		}
	}
	return m;
}
const Matrix operator/(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			m.matrix_[i][j] /= m2.matrix_[i][j];
		}
	}
	return m;
}
const Matrix& Matrix::operator+=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			matrix_[i][j] += m.matrix_[i][j];
		}
	}
	return *this;
}
const Matrix& Matrix::operator-=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			matrix_[i][j] -= m.matrix_[i][j];
		}
	}
	return *this;
}
const Matrix& Matrix::operator*=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			matrix_[i][j] *= m.matrix_[i][j];
		}
	}
	return *this;
}
const Matrix& Matrix::operator/=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (size_t i = 0; i < m.numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			matrix_[i][j] /= m.matrix_[i][j];
		}
	}
	return *this;
}
Matrix Matrix::dot(const Matrix& m) const {
	if (numCols() != m.numRows()) {
		//TODO: throw exception
	}
	Contiguous2DArray<ComplexNumber> result(numRows(), m.numCols());
	#pragma omp parallel for
	for (size_t i = 0; i < numRows(); i++) {
		for (size_t j = 0; j < m.numCols(); j++) {
			for (size_t k = 0; k < numCols(); k++) {
				result[i][j] += matrix_[i][k] * m.matrix_[k][j];
			}
		}
	}
	return Matrix(result, numRows(), m.numCols());
}
Matrix Matrix::augment(const Matrix& m) const {
	if (numRows() != m.numRows()) {
		//TODO: throw exception
	}
	Contiguous2DArray<ComplexNumber> c(numRows(), numCols() + m.numCols());
	#pragma omp parallel for
	for (size_t i = 0; i < numRows(); i++) {
		for (size_t j = 0; j < numCols(); j++) {
			c[i][j] = matrix_[i][j];
		}
		for (size_t j = 0; j < m.numCols(); j++) {
			c[i][j + numCols()] = m.matrix_[i][j];
		}
	}
	return Matrix(c, numRows(), numCols() + m.numCols());
}
const Matrix Matrix::transpose() const {
	Contiguous2DArray<ComplexNumber> result(numCols(), numRows());
	#pragma omp parallel for
	for (size_t i = 0; i < numCols(); i++) {
		for (size_t j = 0; j < numRows(); j++) {
			result[i][j] = matrix_[j][i];
		}
	}
	return Matrix(result, numCols(), numRows());
}
Matrix Matrix::leastSquares(const Matrix& A, const Matrix& b) {
	if (b.numCols() != 1) {
		//TODO: throw exception
	}
	return SquareMatrix(A.transpose().dot(A)).inverse().dot(A.transpose()).dot(b);
}
void Matrix::checkDimensions(const Matrix& m1, const Matrix& m2) {
	if (m1.numRows() != m2.numRows() || m1.numCols() != m2.numCols()) {
		//TODO: throw Exception
	}
}
const Matrix Matrix::conjugateTranspose() const {
	Contiguous2DArray<ComplexNumber> result(numCols(), numCols());
	#pragma omp parallel for
	for (size_t i = 0; i < numCols(); i++) {
		for (size_t j = 0; j < numRows(); j++) {
			result[i][j] = matrix_[j][i].conjugate();
		}
	}
	return Matrix(result, numCols(), numRows());
}
std::ostream& operator<<(std::ostream& os, const Matrix& c) {
	os << "[[" << c.matrix_[0][0];
	for (size_t j = 1; j < c.numCols(); j++) {
		os << ", " << c.matrix_[0][j];
	}
	os << ']';
	for (size_t i = 1; i < c.numRows(); i++) {
		os << ",\n[" << c.matrix_[i][0];
		for (size_t j = 1; j < c.numCols(); j++) {
			os << ", " << c.matrix_[i][j];
		}
		os << ']';
	}
	os << ']';
	return os;
}
std::istream& operator>>(std::istream& is, Matrix& c) {
	//TODO: implement this
	std::string str;
	std::getline(is, str);
	if (str.length() < 5) {
		//TODO: throw exception
		return is;
	}
	c = str;
	/*
	std::vector<std::vector<std::string>> rowVectors;
	std::vector<std::string> vec;
	str.erase(remove_if(str.begin(), str.end(), isspace), str.end()); //remove whitespace
	size_t prev = 2, pos;
	while ((pos = str.find_first_of("[,]", prev)) != std::string::npos) {
		char delim = str[pos];
		if (delim == ',') {
			vec.push_back(str.substr(prev, pos));
		}
		else if (delim == ']' && pos + 1 < str.length()) {
			vec.push_back(str.substr(prev, pos - prev));
			rowVectors.push_back(vec);
			vec.clear();
			pos++;
		}
		prev = pos + 1;
	}
	//initialize the Matrix
	c.rows = rowVectors.size();
	c.cols = rowVectors.at(0).size();
	c.matrix = new ComplexNumber*[c.rows];
	for (size_t i = 0; i < c.rows; i++) {
		c.matrix[i] = new ComplexNumber[c.cols];
		for (size_t j = 0; j < c.cols; j++) {
			rowVectors.at(i).at(j) >> c.matrix[i][j]; //see ComplexNumber::operator>>
		}
	}*/
	return is;
}
size_t Matrix::findNextZeroRow(const size_t& index) const {
	for (size_t i = index; i < numRows(); i++) {
		if (allZeros(i, index + 1, numRows())) { //index+1 b/c findNextNonZeroEntry already checked for zeros in column <index>
			return i;
		}
	}
	return -1; //should not happen
}

Matrix Matrix::rowReduce() const {
	//Usually works except for special cases:
	Matrix result(*this);
	for (size_t j = 0; j < numRows(); j++) {
		size_t nextRow = result.findNextNonZeroEntry(j);
		if (nextRow == numRows()) {
			continue;
		} else if (nextRow == -1) { //the entire column has 0 entries
			nextRow = result.findNextZeroRow(j);
			swap(result.matrix_[nextRow], result.matrix_[j], result.numCols());
			continue;
		} else if (nextRow != j) { //keeps the pivots in the correct rows/columns
			swap(result.matrix_[nextRow], result.matrix_[j], result.numCols());
			nextRow = j;
		}
		//divide row nextRow by the first non-zero entry in that row
		//this results in a "1" in column j
		ComplexNumber divisor = result.matrix_[nextRow][j];
		for (size_t k = j; k < numCols(); k++) {
			result.matrix_[nextRow][k] /= divisor;
		}
		//zero out column j of every row except for nextRow
		for (size_t i = 0; i < numRows(); i++) {
			if (i != nextRow) {
				ComplexNumber multiple = result.matrix_[i][j];
				for (size_t k = j; k < numCols(); k++) {
					result.matrix_[i][k] -= multiple * result.matrix_[nextRow][k];
				}
			}
		}
	}
	return result;
}
Matrix Matrix::verticalPad(const size_t& zeros) const {
	Contiguous2DArray<ComplexNumber> c(numRows() + zeros, numCols());
	copyMatrix(c, matrix_, numRows(), numCols());
	return Matrix(c, numRows() + zeros, numCols());
}
Matrix Matrix::horizontalPad(const size_t& zeros) const {
	Contiguous2DArray<ComplexNumber> c(numRows(), numCols() + zeros);
	copyMatrix(c, matrix_, numRows(), numCols());
	return Matrix(c, numRows(), numCols() + zeros);
}

Matrix Matrix::nullSpace() const { //this function is incomplete
	/* Augment a column of zeros to the right: */
	//Matrix augmented = augment(Matrix(rows, 1)); //not really necessary
	//Matrix reduced = augmented.rowReduce();
	Matrix reduced = rowReduce();
	return reduced; //TODO: complete this function
}
/* Returns true if row i is all zeros from col start to col end. */
bool Matrix::allZeros(const size_t& row, const size_t& start, const size_t& end) const {
	ComplexNumber zero = ComplexNumber(0, 0);
	for (size_t col = start; col < end; col++) {
		if (matrix_[row][col] != zero) {
			return false;
		}
	}
	return true;
}
size_t Matrix::numRows() const {
	return matrix_.numRows();
}
size_t Matrix::numCols() const {
	return matrix_.numCols();
}

//protected functions

/* Returns the ith row. Assumes that derived classes request a valid row. */
ComplexNumber* Matrix::getRow(const size_t& i) const {
	return matrix_[i];
}

/* Returns the row of the next non-zero entry in the specified column, or -1. */
size_t Matrix::findNextNonZeroEntry(const size_t& column) {
	if (column == numRows()) {
		return column; //if there are no more rows
	}
	for (size_t i = column; i < numRows(); i++) {
		if (matrix_[i][column] != ComplexNumber(0, 0)) {
			return i;
		}
	}
	return -1;
}

/** Copies all values from src into dest. */
void Matrix::copyMatrix(Contiguous2DArray<ComplexNumber>& dest, const Contiguous2DArray<ComplexNumber>& src, const size_t& rows, const size_t& cols) const {
	#pragma omp parallel for
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			dest[i][j] = src[i][j];
		}
	}
}

//private functions

/*std::vector<std::string> parseRow(const std::string& str) {

}*/

/** Swaps the elements of an array that contains size elements. */
void Matrix::swap(ComplexNumber* c1, ComplexNumber* c2, const size_t& size)  const {
	ComplexNumber temp;
	for (size_t i = 0; i < size; i++) {
		temp = c1[i];
		c1[i] = c2[i];
		c2[i] = temp;
	}
}
const std::vector<std::vector<std::string>> Matrix::parseRowVectors(const std::string s) const {
	std::string str(s);
	std::vector<std::vector<std::string>> rowVectors;
	std::vector<std::string> vec;
	str.erase(remove_if(str.begin(), str.end(), isspace), str.end()); //remove whitespace
	size_t prev = 2, pos;
	while ((pos = str.find_first_of("[,]", prev)) != std::string::npos) {
		char delim = str[pos];
		if (delim == ',') {
			vec.push_back(str.substr(prev, pos - prev));
		}
		else if (delim == ']' && pos + 1 < str.length()) {
			vec.push_back(str.substr(prev, pos - prev));
			rowVectors.push_back(vec);
			vec.clear();
			pos++;
		}
		prev = pos + 1;
	}
	return rowVectors;
}
void Matrix::initMatrix(const std::vector<std::vector<std::string>>& rowVectors) {
	size_t rows = rowVectors.size();
	size_t cols = rowVectors.at(0).size();
	matrix_ = Contiguous2DArray<ComplexNumber>(rows, cols);
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			rowVectors.at(i).at(j) >> matrix_[i][j]; //see ComplexNumber::operator>>
		}
	}
}
