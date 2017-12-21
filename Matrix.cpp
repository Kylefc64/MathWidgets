#include "Matrix.h"

Matrix::Matrix(const std::string& s) {
	//Assumes str is syntactically correct (for now)
	std::string str(s);
	std::vector<std::vector<std::string>> rowVectors;
	std::vector<std::string> vec;
	str.erase(remove_if(str.begin(), str.end(), isspace), str.end()); //remove whitespace
	size_t prev = 2, pos;
	while ((pos = str.find_first_of("[,]", prev)) != std::string::npos) {
		char delim = str[pos];
		if (delim == ',') {
			vec.push_back(str.substr(prev, pos - prev));
		} else if (delim == ']' && pos + 1 < str.length()) {
			vec.push_back(str.substr(prev, pos - prev));
			rowVectors.push_back(vec);
			vec.clear();
			pos++;
		}
		prev = pos + 1;
	}
	//initialize the Matrix
	rows = rowVectors.size();
	cols = rowVectors.at(0).size();
	matrix = new ComplexNumber*[rows];
	for (size_t i = 0; i < rows; i++) {
		matrix[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			rowVectors.at(i).at(j) >> matrix[i][j]; //see ComplexNumber::operator>>
		}
	}
}
Matrix::Matrix(const int& r, const int& c) :rows(r), cols(c) {
	matrix = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		matrix[i] = new ComplexNumber[cols];
	}
}
Matrix::Matrix(ComplexNumber** m, const int& r, const int& c) :rows(r), cols(c) {
	matrix = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		matrix[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			matrix[i][j] = m[i][j];
		}
	}
}
Matrix::Matrix(const Matrix& m):rows(m.rows), cols(m.cols) {
	matrix = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		matrix[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			matrix[i][j] = m.matrix[i][j];
		}
	}
}
Matrix::Matrix(Matrix&& m) :rows(m.rows), cols(m.cols) {
	matrix = m.matrix;
	m.matrix = nullptr;
}
Matrix::~Matrix() {
	if (matrix) {
		#pragma omp parallel for
		for (int i = 0; i < rows; i++) {
			delete[] matrix[i];
		}
		delete[] matrix;
	}
}
Matrix& Matrix::operator=(const Matrix& m) {
	if (this == &m) {
		return *this;
	}
	if (rows == m.rows && cols == m.cols) {
		//do not reallocate matrix
		#pragma omp parallel for
		for (int i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				matrix[i][j] = m.matrix[i][j];
			}
		}
		return *this;
	}
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		delete[] matrix[i];
	}
	#pragma omp barrier
	delete[] matrix;
	rows = m.rows;
	cols = m.cols;
	matrix = new ComplexNumber*[m.rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		matrix[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			matrix[i][j] = m.matrix[i][j];
		}
	}
	return *this;
}
Matrix& Matrix::operator=(Matrix&& m) {
	//std::cout << "move assignment operator";
	ComplexNumber** temp = m.matrix;
	int r = m.rows, c = m.cols;
	m.matrix = matrix;
	m.rows = rows;
	m.cols = cols;
	matrix = temp;
	rows = r;
	cols = c;
	return *this;
}
ComplexNumber& Matrix::operator()(const int& row, const int& col) {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		//TODO: throw exception
	}
	return matrix[row][col];
}
const Matrix operator+(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			m.matrix[i][j] += m2.matrix[i][j];
		}
	}
	return m;
}
const Matrix operator-(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			m.matrix[i][j] -= m2.matrix[i][j];
		}
	}
	return m;
}
const Matrix operator*(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			m.matrix[i][j] *= m2.matrix[i][j];
		}
	}
	return m;
}
const Matrix operator/(const Matrix& m1, const Matrix& m2) {
	Matrix::checkDimensions(m1, m2);
	Matrix m = Matrix(m1);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			m.matrix[i][j] /= m2.matrix[i][j];
		}
	}
	return m;
}
const Matrix& Matrix::operator+=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			matrix[i][j] += m.matrix[i][j];
		}
	}
	return *this;
}
const Matrix& Matrix::operator-=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			matrix[i][j] -= m.matrix[i][j];
		}
	}
	return *this;
}
const Matrix& Matrix::operator*=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			matrix[i][j] *= m.matrix[i][j];
		}
	}
	return *this;
}
const Matrix& Matrix::operator/=(const Matrix& m) {
	Matrix::checkDimensions(*this, m);
	#pragma omp parallel for
	for (int i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			matrix[i][j] /= m.matrix[i][j];
		}
	}
	return *this;
}
const Matrix Matrix::dot(const Matrix& m) const {
	if (cols != m.rows) {
		//TODO: throw exception
	}
	ComplexNumber** result = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		result[i] = new ComplexNumber[m.cols];
		for (size_t j = 0; j < m.cols; j++) {
			for (size_t k = 0; k < cols; k++) {
				result[i][j] += matrix[i][k] * m.matrix[k][j];
			}
		}
	}
	return Matrix(result, rows, m.cols);
}
const Matrix Matrix::inverse() const {
	if (rows != cols) {
		//TODO: throw exception
	}
	Matrix augmented = augment(identity()).rowReduce();
	/*Matrix augmented = augment(identity());
	for (size_t j = 0; j < cols; j++) {
		int nextRow = augmented.findNextNonZeroEntry(j);
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
		if (allZeros(augmented.matrix[i], 0, 2*cols)) {
			//TODO: throw exception
		}
	}
	//de-augment the matrix
	ComplexNumber** c = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		c[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			c[i][j] = augmented.matrix[i][j + cols];
		}
	}
	return Matrix(c, rows, cols);
}

Matrix Matrix::augment(const Matrix& m) const {
	if (rows != m.rows) {
		//TODO: throw exception
	}
	ComplexNumber** c = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		c[i] = new ComplexNumber[cols + m.cols];
		for (size_t j = 0; j < cols; j++) {
			c[i][j] = matrix[i][j];
		}
		for (size_t j = 0; j < m.cols; j++) {
			c[i][j + cols] = m.matrix[i][j];
		}
	}
	return Matrix(c, rows, cols + m.cols);
}
const Matrix Matrix::identity() const {
	if (rows != cols) {
		//TODO: throw exception
	}
	ComplexNumber** c = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		c[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			if (i == j) {
				c[i][j] = 1;
			}
		}
	}
	return Matrix(c, rows, cols);
}
const Matrix Matrix::identity(const int& size) {
	ComplexNumber** c = new ComplexNumber*[size];
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		c[i] = new ComplexNumber[size];
		for (size_t j = 0; j < size; j++) {
			if (i == j) {
				c[i][j] = 1;
			}
		}
	}
	return Matrix(c, size, size);
}
const Matrix Matrix::transpose() const {
	ComplexNumber** result = new ComplexNumber*[cols];
	#pragma omp parallel for
	for (int i = 0; i < cols; i++) {
		result[i] = new ComplexNumber[rows];
		for (size_t j = 0; j < rows; j++) {
			result[i][j] = matrix[j][i];
		}
	}
	return Matrix(result, cols, rows);
}
Matrix Matrix::solve(const Matrix& A, const Matrix& b) {
	if (b.cols != 1) {
		//TODO: throw exception
	}
	return A.inverse().dot(b);
}
Matrix Matrix::leastSquares(const Matrix& A, const Matrix& b) {
	if (b.cols != 1) {
		//TODO: throw exception
	}
	return A.transpose().dot(A).inverse().dot(A.transpose()).dot(b);
}
void Matrix::checkDimensions(const Matrix& m1, const Matrix& m2) {
	if (m1.rows != m2.rows || m1.cols != m2.cols) {
		//TODO: throw Exception
	}
}
const Matrix Matrix::conjugateTranspose() const {
	ComplexNumber** result = new ComplexNumber*[cols];
	#pragma omp parallel for
	for (int i = 0; i < cols; i++) {
		result[i] = new ComplexNumber[rows];
		for (size_t j = 0; j < rows; j++) {
			result[i][j] = matrix[j][i].conjugate();
		}
	}
	return Matrix(result, cols, rows);
}
std::ostream& operator<<(std::ostream& os, const Matrix& c) {
	os << "[[" << c.matrix[0][0];
	for (size_t j = 1; j < c.cols; j++) {
		os << ", " << c.matrix[0][j];
	}
	os << ']';
	for (size_t i = 1; i < c.rows; i++) {
		os << ",\n[" << c.matrix[i][0];
		for (size_t j = 1; j < c.cols; j++) {
			os << ", " << c.matrix[i][j];
		}
		os << ']';
	}
	os << ']';
	return os;
}
std::istream& operator >> (std::istream& is, Matrix& c) {
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
int Matrix::findNextZeroRow(const int& index) const {
	for (size_t i = index; i < rows; i++) {
		if (allZeros(matrix[i], index + 1, rows)) { //index+1 b/c findNextNonZeroEntry already checked for zeros in column <index>
			return i;
		}
	}
	return -1; //should not happen
}

Matrix Matrix::rowReduce() const {
	//Usually works except for special cases:
	Matrix result(*this);
	for (size_t j = 0; j < rows; j++) {
		int nextRow = result.findNextNonZeroEntry(j);
		if (nextRow == rows) {
			continue;
		} else if (nextRow == -1) { //the entire column has 0 entries
			nextRow = result.findNextZeroRow(j);
			swap(result.matrix[nextRow], result.matrix[j]);
			continue;
		} else if (nextRow != j) { //keeps the pivots in the correct rows/columns
			swap(result.matrix[nextRow], result.matrix[j]);
			nextRow = j;
		}
		//divide row nextRow by the first non-zero entry in that row
		//this results in a "1" in column j
		ComplexNumber divisor = result.matrix[nextRow][j];
		for (size_t k = j; k < cols; k++) {
			result.matrix[nextRow][k] /= divisor;
		}
		//zero out column j of every row except for nextRow
		for (size_t i = 0; i < rows; i++) {
			if (i != nextRow) {
				ComplexNumber multiple = result.matrix[i][j];
				for (size_t k = j; k < cols; k++) {
					result.matrix[i][k] -= multiple * result.matrix[nextRow][k];
				}
			}
		}
	}
	return result;
}
Matrix Matrix::verticalPad(const int& zeros) const {
	ComplexNumber** c = new ComplexNumber*[rows + zeros];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		c[i] = new ComplexNumber[cols];
		for (size_t j = 0; j < cols; j++) {
			c[i][j] = matrix[i][j];
		}
	}
	return Matrix(c, rows + zeros, cols);
}
Matrix Matrix::horizontalPad(const int& zeros) const {
	ComplexNumber** c = new ComplexNumber*[rows];
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		c[i] = new ComplexNumber[cols + zeros];
		for (size_t j = 0; j < cols + zeros; j++) {
			c[i][j] = matrix[i][j];
		}
	}
	return Matrix(c, rows, cols + zeros);
}

Matrix Matrix::nullSpace() const { //this function is incomplete
	/* Augment a column of zeros to the right: */
	//Matrix augmented = augment(Matrix(rows, 1)); //not really necessary
	//Matrix reduced = augmented.rowReduce();
	Matrix reduced = rowReduce();
	return reduced; //TODO: complete this function
}

//private functions

/*std::vector<std::string> parseRow(const std::string& str) {

}*/

int Matrix::findNextNonZeroEntry(const int& column) {
	if (column == rows) {
		return column; //if there are no more rows
	}
	for (int i = column; i < rows; i++) {
		if (matrix[i][column] != ComplexNumber(0, 0)) {
			return i;
		}
	}
	return -1;
}
bool Matrix::allZeros(ComplexNumber* c, const int& start, const int& end) const {
	for (size_t i = start; i < end; i++) {
		if (c[i] != ComplexNumber(0, 0)) {
			return false;
		}
	}
	return true;
}
void Matrix::swap(ComplexNumber*& c1, ComplexNumber*& c2)  const {
	ComplexNumber* temp = c1;
	c1 = c2;
	c2 = temp;
	temp = nullptr;
}