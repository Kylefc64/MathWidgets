/*
TODO:
-Automate testing so that manual inspection is not necessary to verify correctness
-Fix inverse function

This program will perform common matrix operations including, but not limited to:
-Element-wise matrix addition, subtraction, multiplication, division - done
-Matrix dot product calculation - done
-Matrix transposition & conjugate transposition (and overall support for complex-valued matrices) - done
-Matrix inverse calculation - done
-Matrix padding - done
-Ax = b solver - done
-Linear Least Squares Estimation (x = ((aTa)^-1)aTb) - done
-Nullspace calculation
-Eigenvalue and eigenvector calculation
-Matrix diagonalization
-Singular Value Decomposition (SVD)
-Principal Component Analysis (PCA)
-QR Factorizaion(?)
*/
#include <iostream>
#include "Matrix.h"
#include "SquareMatrix.h"
#include <time.h>
#include "Contiguous2DArray.h"
#include <assert.h>

void assertClose(Matrix&, Matrix&, const double&);
void assertEqual(const ComplexNumber&, const ComplexNumber&);
void assertEqual(const Matrix&, const Matrix&);
void testOpenMP();
void testNullSpace();
void testRowReduce();
void testMatrixStringConstructor();
void testConstructor();
void testLeastSquares();
void testSolver();
void testInverseReal();
void testMatrixComplex();
void testMatrix();
void testComplexNumber();
void testContiguous2DArray();

int main() {
	clock_t tStart = clock(); //

	/* un-parallelized (except for testNullSpace) took 34.90 seconds: */
	// need to fix:
	//testOpenMP();
	//testNullSpace();
	testRowReduce();

	// works fine:
	/*testMatrixStringConstructor();
	testConstructor();
	testLeastSquares();
	testSolver();
	testInverseReal();
	testMatrixComplex();
	testMatrix();
	testComplexNumber();
	testContiguous2DArray();*/
	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC); //
	return 0;
}

void assertClose(Matrix& expected, Matrix& actual, const double& tolerance) {
	if (expected.numRows() != actual.numRows() || expected.numCols() != actual.numCols()) {
		std::cout << "--------------------\n"
			<< "Test failed...\n"
			<< "Expected was: " << expected << '\n'
			<< "Actual was: " << actual << '\n'
			<< "--------------------\n";
	}
	double realDiff, imagDiff;
	for (size_t i = 0; i < actual.numRows(); i++) {
		for (size_t j = 0; j < actual.numCols(); j++) {
			realDiff = abs(actual(i, j).real() - expected(i, j).real());
			imagDiff = abs(actual(i, j).imaginary() - expected(i, j).imaginary());
			if (realDiff > tolerance || imagDiff > tolerance) {
				std::cout << "--------------------\n"
					<< "Test failed...\n"
					<< "Expected was: " << expected << '\n'
					<< "Actual was: " << actual << '\n'
					<< "--------------------\n";
				return;
			}
		}
	}
}
void assertEqual(const ComplexNumber& expected, const ComplexNumber& actual) {
	if (expected != actual) {
		std::cout << "--------------------\n"
			<< "Test failed...\n"
			<< "Expected was: " << expected << '\n'
			<< "Actual was: " << actual << '\n'
			<< "--------------------\n";
	}
}
void assertEqual(const Matrix& expected, const Matrix& actual) {
	if (expected != actual) {
		std::cout << "--------------------\n"
			<< "Test failed...\n"
			<< "Expected was: " << expected << '\n'
			<< "Actual was: " << actual << '\n'
			<< "--------------------\n";
	}
}
void testOpenMP() {
	std::cout << "testOpenMP...\n";

	#define MAX_SIZE 3750
	Contiguous2DArray<ComplexNumber> c1(MAX_SIZE, MAX_SIZE);
	Contiguous2DArray<ComplexNumber> c2(MAX_SIZE, MAX_SIZE);
	clock_t tStart = clock(); //start timer
	#pragma omp parallel for
	for (size_t i = 0; i < MAX_SIZE; i++) {
		for (size_t j = 0; j < MAX_SIZE; j++) {
			c1[i][j] = ComplexNumber(i, j);
			c2[i][j] = ComplexNumber(i, j);
		}
	}
	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC); //
	tStart = clock(); //reset timer
	Matrix m1(c1, MAX_SIZE, MAX_SIZE), m2(c2, MAX_SIZE, MAX_SIZE);
	/* With OpenMP, but not YMM, the following 4 operations take 4.26 seconds: */
	/* With OpenMP AND YMM, the following 4 operations take x.xx seconds: */
	Matrix m3 = m1 + m2;
	Matrix m4 = m1 - m2;
	Matrix m5 = m1 * m2;
	Matrix m6 = m1 / m2;
	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC); //
}
void testNullSpace() {
	std::cout << "testNullSpace...\n";

	Matrix m1 = Matrix("[[1,2,3],[0,0,3],[1,2,3]]");
	std::cout << m1 << "\n\n";
	Matrix m2 = m1.nullSpace();
	std::cout << m2 << "\n\n";

	m1 = Matrix("[[1,2,3,4],[1,2,3,4],[1,2,3,4],[0,0,7,9]]");
	std::cout << m1 << "\n\n";
	m2 = m1.nullSpace();
	std::cout << m2 << "\n\n";
}
void testRowReduce() {
	std::cout << "testRowReduce...\n";

	Matrix m1 = Matrix("[[1,2,3],[0, 0, 3],[1, 2, 3]]");

	Matrix m("[[1, 1, 1],[2, 2, 2],[3,4,5]]");
	Matrix reduced = m.rowReduce();

	assertEqual(Matrix("[[1, 2, 0], [0, 0, 0], [0, 0, 1]]"), m1.rowReduce());
	assertEqual(Matrix("[[1, 0, -1], [0, 1, 2], [0, 0, 0]]"), m.rowReduce());
}
void testMatrixStringConstructor() {
	std::cout << "testMatrixStringConstructor...\n";

	Matrix m1("[   [1  ,  2,3]	   ,[4,   5,   6  ],   [   7,8 ,    9]]");

	assertEqual(Matrix("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]"), m1);
	Matrix m2("    [[         2]          ]   ");

	assertEqual(Matrix("[[2]]"), m2);
	m2 = Matrix("  [  [1 +   2 j ,2-  3  j   ],[  - 7j   ,-   8  ]]  ");

	assertEqual(Matrix("[[1+2j, 2-3j], [-7j, -8]]"), m2);
}
void testConstructor() {
	std::cout << "testConstructor...\n";

	Contiguous2DArray<ComplexNumber> c1(2, 2);
	c1[0][0] = ComplexNumber(1, 0);
	c1[0][1] = ComplexNumber(2, 0);

	c1[1][0] = ComplexNumber(3, 0);
	c1[1][1] = ComplexNumber(4, 0);

	Contiguous2DArray<ComplexNumber> c2(2, 2);
	c2[0][0] = ComplexNumber(5, 0);
	c2[0][1] = ComplexNumber(6, 0);

	c2[1][0] = ComplexNumber(7, 0);
	c2[1][1] = ComplexNumber(8, 0);

	Matrix m1 = Matrix(c1, 2, 2);
	Matrix m2 = Matrix(c2, 2, 2);

	assertEqual(Matrix("[[1, 2], [3, 4]]"), m1);
	assertEqual(Matrix("[[5, 6], [7, 8]]"), m2);

	m1 = m2;

	assertEqual(m1, m2);
}
void testLeastSquares() { //answer should be: [[1.0507],[-0.143254]]
	std::cout << "testLeastSquares...\n";

	Contiguous2DArray<ComplexNumber> a(6, 2);

	a[0][0] = ComplexNumber(1.2, 0);
	a[0][1] = ComplexNumber(1, 0);

	a[1][0] = ComplexNumber(2.3, 0);
	a[1][1] = ComplexNumber(1, 0);

	a[2][0] = ComplexNumber(3.0, 0);
	a[2][1] = ComplexNumber(1, 0);

	a[3][0] = ComplexNumber(3.8, 0);
	a[3][1] = ComplexNumber(1, 0);

	a[4][0] = ComplexNumber(4.7, 0);
	a[4][1] = ComplexNumber(1, 0);

	a[5][0] = ComplexNumber(5.9, 0);
	a[5][1] = ComplexNumber(1, 0);

	Contiguous2DArray<ComplexNumber> b(6, 1);

	b[0][0] = ComplexNumber(1.1, 0);

	b[1][0] = ComplexNumber(2.1, 0);

	b[2][0] = ComplexNumber(3.1, 0);

	b[3][0] = ComplexNumber(4.0, 0);

	b[4][0] = ComplexNumber(4.9, 0);

	b[5][0] = ComplexNumber(5.9, 0);

	assertClose(Matrix("[[1.0507], [-0.143254]]"), Matrix::leastSquares(Matrix(a, 6, 2), Matrix(b, 6, 1)), 0.000005);
}
void testSolver() {
	std::cout << "testSolver...\n";

	//x = [2, 3, 5]
	Contiguous2DArray<ComplexNumber> a(3, 3); //A

	a[0][0] = ComplexNumber(1, 0);
	a[0][1] = ComplexNumber(2, 0);
	a[0][2] = ComplexNumber(3, 0);

	a[1][0] = ComplexNumber(4, 0);
	a[1][1] = ComplexNumber(5, 0);
	a[1][2] = ComplexNumber(6, 0);

	a[2][0] = ComplexNumber(-2, 0);
	a[2][1] = ComplexNumber(-3, 0);
	a[2][2] = ComplexNumber(7, 0);

	Contiguous2DArray<ComplexNumber> b(3, 1); //b

	b[0][0] = ComplexNumber(23, 0);

	b[1][0] = ComplexNumber(53, 0);

	b[2][0] = ComplexNumber(22, 0);

	SquareMatrix A(a, 3);
	Matrix bVec(b, 3, 1);

	assertEqual(Matrix("[[23], [53], [22]]"), bVec);
	assertClose(Matrix("[[-1.60606, 0.69697, 0.0909091], [1.21212, -0.393939, -0.181818], [0.0606061, 0.030303, 0.0909091]]"), A.inverse(), 0.000002);
	assertClose(Matrix("[[2], [3], [5]]"), SquareMatrix::solve(A, bVec), 0.00000000000002);
}
void testInverseReal() {
	std::cout << "testInverseReal...\n";

	SquareMatrix m1 = SquareMatrix("[[1,2,3],[0, 0, 3],[0, 5, 0]]");

	SquareMatrix m2 = m1.inverse();

	assertEqual(Matrix("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"), m1.rowReduce());
	assertClose(Matrix("[[1, -1, -0.4], [0, 0, 0.2], [0, 0.3333333, 0]]"), m1.inverse(), 0.00000002);
	assertEqual(Matrix("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"), m2.dot(m1));

	/*
	ComplexNumber** c = Matrix::allocateContiguousMatrix(3, 3);

	c[0][0] = ComplexNumber(0, 0);
	c[0][1] = ComplexNumber(2, 0);
	c[0][2] = ComplexNumber(3, 0);

	c[1][0] = ComplexNumber(4, 0);
	c[1][1] = ComplexNumber(5, 0);
	c[1][2] = ComplexNumber(6, 0);

	c[2][0] = ComplexNumber(9, 0);
	c[2][1] = ComplexNumber(8, 0);
	c[2][2] = ComplexNumber(5, 0);

	Matrix m(c, 3, 3);

	std::cout << m << "\n\n"
		<< m.inverse() << "\n\n";
		*/
}
void testMatrixComplex() {
	std::cout << "testMatrixComplex...\n";

	Contiguous2DArray<ComplexNumber> c1(3, 5); //3x5 matrix

	c1[0][0] = ComplexNumber(1, 2);
	c1[0][1] = ComplexNumber(2, 3);
	c1[0][2] = ComplexNumber(3, 4);
	c1[0][3] = ComplexNumber(4, 5);
	c1[0][4] = ComplexNumber(5, 6);

	c1[1][0] = ComplexNumber(6, 7);
	c1[1][1] = ComplexNumber(7, 8);
	c1[1][2] = ComplexNumber(8, 9);
	c1[1][3] = ComplexNumber(9, 10);
	c1[1][4] = ComplexNumber(10, 11);

	c1[2][0] = ComplexNumber(11, 12);
	c1[2][1] = ComplexNumber(12, 13);
	c1[2][2] = ComplexNumber(13, 14);
	c1[2][3] = ComplexNumber(14, 15);
	c1[2][4] = ComplexNumber(15, 16);

	Matrix m1 = Matrix(c1, 3, 5);

	assertEqual(Matrix("[[1-2j, 6-7j, 11-12j], [2-3j, 7-8j, 12-13j], [3-4j, 8-9j, 13-14j], [ 4-5j, 9-10j, 14-15j], [5-6j, 10-11j, 15-16j]]"), m1.conjugateTranspose());
}
void testMatrix() {
	std::cout << "testMatrix...\n";

	Contiguous2DArray<ComplexNumber> c1(3, 4); //3x4 matrix

	c1[0][0] = ComplexNumber(1, 0);
	c1[0][1] = ComplexNumber(2, 0);
	c1[0][2] = ComplexNumber(3, 0);
	c1[0][3] = ComplexNumber(4, 0);

	c1[1][0] = ComplexNumber(5, 0);
	c1[1][1] = ComplexNumber(6, 0);
	c1[1][2] = ComplexNumber(7, 0);
	c1[1][3] = ComplexNumber(8, 0);

	c1[2][0] = ComplexNumber(9, 0);
	c1[2][1] = ComplexNumber(10, 0);
	c1[2][2] = ComplexNumber(11, 0);
	c1[2][3] = ComplexNumber(12, 0);

	Contiguous2DArray<ComplexNumber> c2(4, 3); //4x3 matrix

	c2[0][0] = ComplexNumber(9, 0);
	c2[0][1] = ComplexNumber(8, 0);
	c2[0][2] = ComplexNumber(7, 0);

	c2[1][0] = ComplexNumber(5, 0);
	c2[1][1] = ComplexNumber(4, 0);
	c2[1][2] = ComplexNumber(3, 0);

	c2[2][0] = ComplexNumber(1, 0);
	c2[2][1] = ComplexNumber(12, 0);
	c2[2][2] = ComplexNumber(11, 0);

	c2[3][0] = ComplexNumber(6, 0);
	c2[3][1] = ComplexNumber(2, 0);
	c2[3][2] = ComplexNumber(10, 0);

	Matrix m1 = Matrix(c1, 3, 4);
	Matrix m2 = Matrix(c2, 4, 3);

	Matrix sum = m1 + m1;
	Matrix diff = m1 - m1;
	Matrix product = m1 * m1;
	Matrix quotient = m1 / m1;
	Matrix dot = m1.dot(m2);
	Matrix transpose = m1.transpose();
	Matrix conjTranspose = m1.conjugateTranspose();

	assertEqual(Matrix("[[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24]]"), sum);
	assertEqual(Matrix("[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"), diff);
	assertEqual(Matrix("[[1, 4, 9, 16], [25, 36, 49, 64], [81, 100, 121, 144]]"), product);
	assertEqual(Matrix("[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]"), quotient);
	assertEqual(Matrix("[[46, 60, 86], [130, 164, 210], [214, 268, 334]]"), dot);
	assertEqual(Matrix("[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]"), transpose);
	assertEqual(Matrix("[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]"), conjTranspose);
}
void testComplexNumber() {
	std::cout << "testComplexNumber...\n";

	ComplexNumber c1 = ComplexNumber(3, 4);
	ComplexNumber c2 = ComplexNumber(5, 7);
	ComplexNumber c3 = ComplexNumber(5, 5);
	c3 += ComplexNumber(6, 9);

	assertEqual(ComplexNumber(11, 14), c3);
	assertEqual(ComplexNumber(8, 11), (c1 + c2));
	assertEqual(ComplexNumber(-2, -3), (c1 - c2));
	assertEqual(ComplexNumber(-13, 41), (c1 * c2));
	assertEqual(ComplexNumber((double)43 / 74, (double)-1 / 74), (c1 / c2));
}
void testContiguous2DArray() {
	std::cout << "testcontiguous2DArray...\n";

	Contiguous2DArray<int> intArr(2, 2);
	intArr[0][0] = 1;
	intArr[0][1] = 2;
	intArr[1][0] = 3;
	intArr[1][1] = 4;

	Contiguous2DArray<ComplexNumber> complexArr(2, 2);
	complexArr[0][0] = ComplexNumber(1, 0);
	complexArr[0][1] = ComplexNumber(2, 0);
	complexArr[1][0] = ComplexNumber(3, 0);
	complexArr[1][1] = ComplexNumber(4, 0);

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			assert(intArr[i][j] == complexArr[i][j].real());
		}
	}
}
