#include "ComplexNumber.h"

double ComplexNumber::magnitude() const {
	return sqrt(real_*real_ + imaginary_*imaginary_);
}
const ComplexNumber operator+(const ComplexNumber& c1, const ComplexNumber& c2) {
	return ComplexNumber(c1.real_ + c2.real_, c1.imaginary_ + c2.imaginary_);
}
const ComplexNumber operator-(const ComplexNumber& c1, const ComplexNumber& c2) {
	return ComplexNumber(c1.real_ - c2.real_, c1.imaginary_ - c2.imaginary_);
}
const ComplexNumber operator*(const ComplexNumber& c1, const ComplexNumber& c2) {
	return ComplexNumber(c1.real_*c2.real_ - c1.imaginary_*c2.imaginary_, c1.real_*c2.imaginary_ + c1.imaginary_*c2.real_);
}
const ComplexNumber operator/(const ComplexNumber& c1, const ComplexNumber& c2) {
	ComplexNumber conj = c2.conjugate();
	ComplexNumber numerator = c1*conj;
	ComplexNumber denominator = c2*conj; //imaginary part should be 0
	return ComplexNumber(numerator.real_ / denominator.real_, numerator.imaginary_ / denominator.real_);
}
const ComplexNumber& ComplexNumber::operator+=(const ComplexNumber& c) {
	real_ += c.real_;
	imaginary_ += c.imaginary_;
	return *this;
}
const ComplexNumber& ComplexNumber::operator-=(const ComplexNumber& c) {
	real_ -= c.real_;
	imaginary_ -= c.imaginary_;
	return *this;
}
const ComplexNumber& ComplexNumber::operator*=(const ComplexNumber& c) {
	double r = real_*c.real_ - imaginary_*c.imaginary_;
	imaginary_ = real_*c.imaginary_ + imaginary_*c.real_;
	real_ = r;
	return *this;
}
const ComplexNumber& ComplexNumber::operator/=(const ComplexNumber& c) {
	ComplexNumber conj = c.conjugate();
	ComplexNumber numerator = *this*conj;
	ComplexNumber denominator = c*conj;
	real_ = numerator.real_ / denominator.real_;
	imaginary_ = numerator.imaginary_ / denominator.real_;
	return *this;
}
std::ostream& operator<<(std::ostream& os, const ComplexNumber& c) {
	if (c.real_ == 0 && c.imaginary_ != 0) {
		os << c.imaginary_ << 'j';
	} else {
		os << c.real_;
		if (c.imaginary_ > 0) {
			os << '+' << c.imaginary_ << 'j';
		} else if (c.imaginary_ < 0) {
			os << c.imaginary_ << 'j';
		}
	}
	return os;
}
std::istream& operator>>(std::istream& is, ComplexNumber& c) {
	//TODO: should accept CNs of the form a + bj
	is >> c.real_ >> c.imaginary_;
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
			c.real_ = strtof(r.c_str(), 0);
		} else if (minusPos != std::string::npos) {
			r = str.substr(0, minusPos + 1);
			i = str.substr(minusPos, str.length() - minusPos);
			c.real_ = strtof(r.c_str(), 0);
		} else {
			i = str.substr(0, str.length() - 1);
		}
		c.imaginary_ = strtof(i.c_str(), 0);
	} else {
		c.real_ = strtof(str.c_str(), 0);
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
	return lhs.real_ == rhs.real_ && lhs.imaginary_ == rhs.imaginary_;
}
const bool operator!=(const ComplexNumber& lhs, const ComplexNumber& rhs) {
	return lhs.real_ != rhs.real_ || lhs.imaginary_ != rhs.imaginary_;
}
ComplexNumber& ComplexNumber::operator=(const ComplexNumber& other) {
	real_ = other.real_;
	imaginary_ = other.imaginary_;
	return *this;
}
ComplexNumber& ComplexNumber::operator=(const double& r) {
	real_ = r;
	imaginary_ = 0;
	return *this;
}
