/*
This class represents a complex number of the form a + bj
and supports common arithmetic operations between complex numbers
*/
//#pragma once
#ifndef COMPLEXNUMBER_H
#define COMPLEXNUMBER_H
#include <ostream>
#include <istream>
#include <iostream>
#include <algorithm>

class ComplexNumber {
public:
	ComplexNumber() :real_(0), imaginary_(0) {}
	ComplexNumber(double r, double i) :real_(r), imaginary_(i) {}
	ComplexNumber(const ComplexNumber& c) :real_(c.real_), imaginary_(c.imaginary_) {}
	//ComplexNumber(const ComplexNumber&& c) :real(c.real), imaginary(c.imaginary) { std::cout << "Move constructor\n"; }
	~ComplexNumber() {}
	ComplexNumber conjugate() const { return ComplexNumber(real_, -imaginary_); }
	double magnitude() const;
	friend const ComplexNumber operator+(const ComplexNumber&, const ComplexNumber&); //Returns the addition of two complex numbers
	friend const ComplexNumber operator-(const ComplexNumber&, const ComplexNumber&); //Returns the difference of two complex numbers
	friend const ComplexNumber operator*(const ComplexNumber&, const ComplexNumber&); //Returns the product of two complex numbers
	friend const ComplexNumber operator/(const ComplexNumber&, const ComplexNumber&); //Returns the product of two complex numbers
	friend const bool operator<(const ComplexNumber&, const ComplexNumber&); //compares based on magnitude
	friend const bool operator>(const ComplexNumber&, const ComplexNumber&); //""
	friend const bool operator<=(const ComplexNumber&, const ComplexNumber&); //""
	friend const bool operator>=(const ComplexNumber&, const ComplexNumber&); //""
	friend const bool operator==(const ComplexNumber&, const ComplexNumber&); //""
	friend const bool operator!=(const ComplexNumber&, const ComplexNumber&); //""
	const ComplexNumber& operator+=(const ComplexNumber&);
	const ComplexNumber& operator-=(const ComplexNumber&);
	const ComplexNumber& operator*=(const ComplexNumber&);
	const ComplexNumber& operator/=(const ComplexNumber&);
	ComplexNumber& operator=(const ComplexNumber&);
	ComplexNumber& operator=(const double&);
	friend std::ostream& operator<<(std::ostream&, const ComplexNumber&); //Inserts the contents of a ComplexNumber into an ostream
	friend std::istream& operator>>(std::istream&, ComplexNumber&); //Extracts ComplexNumber data from an istream and inserts it into a ComplexNumber object
	friend ComplexNumber& operator >> (const std::string&, ComplexNumber&);
	double real() const { return real_; }
	double imaginary() const { return imaginary_; }
private:
	double real_;
	double imaginary_;
};
#endif
