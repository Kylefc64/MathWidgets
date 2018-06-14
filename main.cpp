/*
TODO:
-Use size_t instead of int for rows and cols
-Represent matrices as a flat contiguous array instead of 2D non-contiguous arrays
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

int main() {
	clock_t tStart = clock(); //

	/* un-parallelized (except for testNullSpace) took 34.90 seconds: */
	testOpenMP();
	testNullSpace();
	testRowReduce();
	//testMatrixStringConstructor();
	//testConstructor();
	//testLeastSquares();
	//testSolver();
	//testInverseReal();
	//testMatrixComplex();
	//testMatrix();
	//testComplexNumber();
	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC); //
	return 0;
}

void testOpenMP() {
	#define MAX_SIZE 3750
	ComplexNumber **c1 = new ComplexNumber*[MAX_SIZE];
	ComplexNumber **c2 = new ComplexNumber*[MAX_SIZE];
	clock_t tStart = clock(); //start timer
	#pragma omp parallel for
	for (int i = 0; i < MAX_SIZE; i++) {
		c1[i] = new ComplexNumber[MAX_SIZE];
		c2[i] = new ComplexNumber[MAX_SIZE];
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
	Matrix m1 = Matrix("[[1,2,3],[0, 0, 3],[1, 2, 3]]");
	std::cout << m1.rowReduce() << "\n\n";

	Matrix m("[[1, 1, 1],[2, 2, 2],[3,4,5]]");
	Matrix reduced = m.rowReduce();
	std::cout << reduced << "\n\n";
}
void testMatrixStringConstructor() {
	Matrix m1("[   [1  ,  2,3]	   ,[4,   5,   6  ],   [   7,8 ,    9]]");
	std::cout << m1 << "\n\n";
	Matrix m2("[[2]]");
	std::cout << m2 << "\n\n";
	//std::cout << "Enter a matrix:\n";
	//std::cin >> m2;
	//std::cout << m2 << "\n\n";
	m2 = Matrix("[[1+2j,2-3j],[-7j,-8]]");
	std::cout << m2 << "\n\n";
}
void testConstructor() {
	ComplexNumber** c1 = new ComplexNumber*[2];
	c1[0] = new ComplexNumber[2];
	c1[0][0] = ComplexNumber(1, 0);
	c1[0][1] = ComplexNumber(2, 0);
	c1[1] = new ComplexNumber[2];
	c1[1][0] = ComplexNumber(3, 0);
	c1[1][1] = ComplexNumber(4, 0);

	ComplexNumber** c2 = new ComplexNumber*[2];
	c2[0] = new ComplexNumber[2];
	c2[0][0] = ComplexNumber(5, 0);
	c2[0][1] = ComplexNumber(6, 0);
	c2[1] = new ComplexNumber[2];
	c2[1][0] = ComplexNumber(7, 0);
	c2[1][1] = ComplexNumber(8, 0);

	Matrix m1 = Matrix(c1, 2, 2);
	Matrix m2 = Matrix(c2, 2, 2);
	std::cout << m1 << "\n\n" << m2 << "\n\n";

	m1 = m2;
	std::cout << m1 << "\n\n";
}
void testLeastSquares() { //answer should be: [[1.0507],[-0.143254]]
	ComplexNumber** a = new ComplexNumber*[6];
	a[0] = new ComplexNumber[2];
	a[0][0] = ComplexNumber(1.2, 0);
	a[0][1] = ComplexNumber(1, 0);
	a[1] = new ComplexNumber[2];
	a[1][0] = ComplexNumber(2.3, 0);
	a[1][1] = ComplexNumber(1, 0);
	a[2] = new ComplexNumber[2];
	a[2][0] = ComplexNumber(3.0, 0);
	a[2][1] = ComplexNumber(1, 0);
	a[3] = new ComplexNumber[2];
	a[3][0] = ComplexNumber(3.8, 0);
	a[3][1] = ComplexNumber(1, 0);
	a[4] = new ComplexNumber[2];
	a[4][0] = ComplexNumber(4.7, 0);
	a[4][1] = ComplexNumber(1, 0);
	a[5] = new ComplexNumber[2];
	a[5][0] = ComplexNumber(5.9, 0);
	a[5][1] = ComplexNumber(1, 0);

	ComplexNumber** b = new ComplexNumber*[6];
	b[0] = new ComplexNumber[1];
	b[0][0] = ComplexNumber(1.1, 0);
	b[1] = new ComplexNumber[1];
	b[1][0] = ComplexNumber(2.1, 0);
	b[2] = new ComplexNumber[1];
	b[2][0] = ComplexNumber(3.1, 0);
	b[3] = new ComplexNumber[1];
	b[3][0] = ComplexNumber(4.0, 0);
	b[4] = new ComplexNumber[1];
	b[4][0] = ComplexNumber(4.9, 0);
	b[5] = new ComplexNumber[1];
	b[5][0] = ComplexNumber(5.9, 0);

	std::cout << Matrix::leastSquares(Matrix(a, 6, 2), Matrix(b, 6, 1)) << "\n\n";
}
void testSolver() {
	//x = [2, 3, 5]
	ComplexNumber** a = new ComplexNumber*[3]; //A
	a[0] = new ComplexNumber[3];
	a[0][0] = ComplexNumber(1, 0);
	a[0][1] = ComplexNumber(2, 0);
	a[0][2] = ComplexNumber(3, 0);
	a[1] = new ComplexNumber[3];
	a[1][0] = ComplexNumber(4, 0);
	a[1][1] = ComplexNumber(5, 0);
	a[1][2] = ComplexNumber(6, 0);
	a[2] = new ComplexNumber[3];
	a[2][0] = ComplexNumber(-2, 0);
	a[2][1] = ComplexNumber(-3, 0);
	a[2][2] = ComplexNumber(7, 0);

	ComplexNumber** b = new ComplexNumber*[3]; //b
	b[0] = new ComplexNumber[1];
	b[0][0] = ComplexNumber(23, 0);
	b[1] = new ComplexNumber[1];
	b[1][0] = ComplexNumber(53, 0);
	b[2] = new ComplexNumber[1];
	b[2][0] = ComplexNumber(22, 0);

	SquareMatrix A(a, 3);
	Matrix bVec(b, 3, 1);

	std::cout << bVec << "\n\n";

	std::cout << A.inverse() << "\n\n";

	std::cout << Matrix::solve(A, bVec) << "\n\n";
}
void testInverseReal() {
	SquareMatrix m1 = SquareMatrix("[[1,2,3],[0, 0, 3],[0, 5, 0]]");
	std::cout << m1.rowReduce() << "\n\n";
	SquareMatrix m2 = m1.inverse();
	std::cout << m2 << "\n\n";
	std::cout << m2.dot(m1) << "\n\n";

	/*
	ComplexNumber** c = new ComplexNumber*[3];
	c[0] = new ComplexNumber[3];
	c[0][0] = ComplexNumber(0, 0);
	c[0][1] = ComplexNumber(2, 0);
	c[0][2] = ComplexNumber(3, 0);
	c[1] = new ComplexNumber[3];
	c[1][0] = ComplexNumber(4, 0);
	c[1][1] = ComplexNumber(5, 0);
	c[1][2] = ComplexNumber(6, 0);
	c[2] = new ComplexNumber[3];
	c[2][0] = ComplexNumber(9, 0);
	c[2][1] = ComplexNumber(8, 0);
	c[2][2] = ComplexNumber(5, 0);

	Matrix m(c, 3, 3);

	std::cout << m << "\n\n"
		<< m.inverse() << "\n\n";
		*/
}
void testMatrixComplex() {
	ComplexNumber** c1 = new ComplexNumber*[3]; //3x5 matrix
	c1[0] = new ComplexNumber[5];
	c1[0][0] = ComplexNumber(1, 2);
	c1[0][1] = ComplexNumber(2, 3);
	c1[0][2] = ComplexNumber(3, 4);
	c1[0][3] = ComplexNumber(4, 5);
	c1[0][4] = ComplexNumber(5, 6);
	c1[1] = new ComplexNumber[5];
	c1[1][0] = ComplexNumber(6, 7);
	c1[1][1] = ComplexNumber(7, 8);
	c1[1][2] = ComplexNumber(8, 9);
	c1[1][3] = ComplexNumber(9, 10);
	c1[1][4] = ComplexNumber(10, 11);
	c1[2] = new ComplexNumber[5];
	c1[2][0] = ComplexNumber(11, 12);
	c1[2][1] = ComplexNumber(12, 13);
	c1[2][2] = ComplexNumber(13, 14);
	c1[2][3] = ComplexNumber(14, 15);
	c1[2][4] = ComplexNumber(15, 16);

	Matrix m1 = Matrix(c1, 3, 5);
	std::cout<< m1 << "\n\n" << m1.conjugateTranspose() << "\n\n";
}
void testMatrix() {
	ComplexNumber** c1 = new ComplexNumber*[3]; //3x4 matrix
	c1[0] = new ComplexNumber[4];
	c1[0][0] = ComplexNumber(1, 0);
	c1[0][1] = ComplexNumber(2, 0);
	c1[0][2] = ComplexNumber(3, 0);
	c1[0][3] = ComplexNumber(4, 0);
	c1[1] = new ComplexNumber[4];
	c1[1][0] = ComplexNumber(5, 0);
	c1[1][1] = ComplexNumber(6, 0);
	c1[1][2] = ComplexNumber(7, 0);
	c1[1][3] = ComplexNumber(8, 0);
	c1[2] = new ComplexNumber[4];
	c1[2][0] = ComplexNumber(9, 0);
	c1[2][1] = ComplexNumber(10, 0);
	c1[2][2] = ComplexNumber(11, 0);
	c1[2][3] = ComplexNumber(12, 0);

	ComplexNumber** c2 = new ComplexNumber*[4]; //4x3 matrix
	c2[0] = new ComplexNumber[3];
	c2[0][0] = ComplexNumber(9, 0);
	c2[0][1] = ComplexNumber(8, 0);
	c2[0][2] = ComplexNumber(7, 0);
	c2[1] = new ComplexNumber[3];
	c2[1][0] = ComplexNumber(5, 0);
	c2[1][1] = ComplexNumber(4, 0);
	c2[1][2] = ComplexNumber(3, 0);
	c2[2] = new ComplexNumber[3];
	c2[2][0] = ComplexNumber(1, 0);
	c2[2][1] = ComplexNumber(12, 0);
	c2[2][2] = ComplexNumber(11, 0);
	c2[3] = new ComplexNumber[3];
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

	std::cout << "m1 + m1 = " << sum << "\n\n"
		<< "m1 - m1 = " << diff << "\n\n"
		<< "m1 * m1 = " << product << "\n\n"
		<< "m1 / m1 = " << quotient << "\n\n"
		<< "m2 = " << m2 << "\n\n"
		<< "m1.dot(m2) = " << dot << "\n\n"
		<< "m1 = " << m1 << "\n\n"
		<< "m1.transpose() = " << transpose << "\n\n"
		<< "m1.conjugateTranspose() = " << conjTranspose << std::endl;
}
void testComplexNumber() {
	ComplexNumber c1 = ComplexNumber(3, 4);
	ComplexNumber c2 = ComplexNumber(5, 7);
	ComplexNumber c3 = ComplexNumber(5, 5);
	c3 += ComplexNumber(6, 9);
	std::cout << c1 + c2 << '\n'
		<< c1 - c2 << '\n'
		<< c1 * c2 << '\n'
		<< c1 / c2 << std::endl;
}