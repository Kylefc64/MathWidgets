#include "ComplexNumber.h"

double ComplexNumber::magnitude() const {
	return sqrt(real*real + imaginary*imaginary);
}
const ComplexNumber operator+(const ComplexNumber& c1, const ComplexNumber& c2) {
	return ComplexNumber(c1.real + c2.real, c1.imaginary + c2.imaginary);
}
const ComplexNumber operator-(const ComplexNumber& c1, const ComplexNumber& c2) {
	return ComplexNumber(c1.real - c2.real, c1.imaginary - c2.imaginary);
}
const ComplexNumber operator*(const ComplexNumber& c1, const ComplexNumber& c2) {
	return ComplexNumber(c1.real*c2.real - c1.imaginary*c2.imaginary, c1.real*c2.imaginary + c1.imaginary*c2.real);
}
const ComplexNumber operator/(const ComplexNumber& c1, const ComplexNumber& c2) {
	ComplexNumber conj = c2.conjugate();
	ComplexNumber numerator = c1*conj;
	ComplexNumber denominator = c2*conj; //imaginary part should be 0
	return ComplexNumber(numerator.real / denominator.real, numerator.imaginary / denominator.real);
}
const ComplexNumber& ComplexNumber::operator+=(const ComplexNumber& c) {
	real += c.real;
	imaginary += c.imaginary;
	return *this;
}
const ComplexNumber& ComplexNumber::operator-=(const ComplexNumber& c) {
	real -= c.real;
	imaginary -= c.imaginary;
	return *this;
}
const ComplexNumber& ComplexNumber::operator*=(const ComplexNumber& c) {
	double r = real*c.real - imaginary*c.imaginary;
	imaginary = real*c.imaginary + imaginary*c.real;
	real = r;
	return *this;
}
const ComplexNumber& ComplexNumber::operator/=(const ComplexNumber& c) {
	ComplexNumber conj = c.conjugate();
	ComplexNumber numerator = *this*conj;
	ComplexNumber denominator = c*conj;
	real = numerator.real / denominator.real;
	imaginary = numerator.imaginary / denominator.real;
	return *this;
}
std::ostream& operator<<(std::ostream& os, const ComplexNumber& c) {
	if (c.real == 0 && c.imaginary != 0) {
		os << c.imaginary << 'j';
	} else {
		os << c.real;
		if (c.imaginary > 0) {
			os << '+' << c.imaginary << 'j';
		} else if (c.imaginary < 0) {
			os << c.imaginary << 'j';
		}
	}
	return os;
}
std::istream& operator>>(std::istream& is, ComplexNumber& c) {
	//TODO: should accept CNs of the form a + bj
	is >> c.real >> c.imaginary;
	return is;
}
ComplexNumber& operator>>(const std::string& s, ComplexNumber& c) {
	std::string str(s);
	str.erase(remove_if(str.begin(), str.end(), isspace), str.end()); //remove whitespace
	if (str[str.length() - 1] == 'j') {
		size_t plusPos = str.find('+');
		size_t minusPos = str.find('-');
		std::string r, i;
		if (plusPos != std::string::npos) {
			r = str.substr(0, plusPos + 1);
			i = str.substr(plusPos, str.length() - plusPos);
			c.real = strtof(r.c_str(), 0);
		} else if (minusPos != std::string::npos) {
			r = str.substr(0, minusPos + 1);
			i = str.substr(minusPos, str.length() - minusPos);
			c.real = strtof(r.c_str(), 0);
		} else {
			i = str.substr(0, str.length() - 1);
		}
		c.imaginary = strtof(i.c_str(), 0);
	} else {
		c.real = strtof(str.c_str(), 0);
	}
	return c;
}
const bool operator<(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.magnitude() < rhs.magnitude();
}
const bool operator>(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.magnitude() > rhs.magnitude();
}
const bool operator<=(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.magnitude() <= rhs.magnitude();
}
const bool operator>=(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.magnitude() >= rhs.magnitude();
}
const bool operator==(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.real == rhs.real && lhs.imaginary == rhs.imaginary;
}
const bool operator!=(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.real != rhs.real || lhs.imaginary != rhs.imaginary;
}
ComplexNumber& ComplexNumber::operator=(const ComplexNumber& other) {
	real = other.real;
	imaginary = other.imaginary;
	return *this;
}
ComplexNumber& ComplexNumber::operator=(const double& r) {
	real = r;
	imaginary = 0;
	return *this;
}