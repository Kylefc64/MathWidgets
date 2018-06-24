#ifndef CONTIGUOUS_2D_ARRAY
#define CONTIGUOUS_2D_ARRAY
#include <omp.h>

/** A wrapper class for dynamically allocated 2D arrays, used as the 
	underlying implementation for the Matrix class. */
template <typename T>
class Contiguous2DArray {
public:
	Contiguous2DArray(const size_t&, const size_t&);
	Contiguous2DArray(Contiguous2DArray<T>&&);
	~Contiguous2DArray();
	T* operator[](const size_t&) const;
	size_t numRows() const { return rows_; }
	size_t numCols() const { return cols_; }
	Contiguous2DArray<T>& operator=(Contiguous2DArray<T>&&);
	void destroyArray();
private:
	size_t rows_;
	size_t cols_;
	T** arr_;
};

template <typename T>
Contiguous2DArray<T>::Contiguous2DArray(const size_t& rows, const size_t& cols) : rows_(rows), cols_(cols) {
	if (rows > 0 && cols > 0) {
		arr_ = new T*[rows_];
		arr_[0] = new T[rows_*cols_];
		#pragma omp parallel for
		for (size_t i = 0; i < rows_; ++i) {
			arr_[i] = &arr_[0][i*cols_];
		}
	} else {
		arr_ = nullptr;
	}
}

template <typename T>
Contiguous2DArray<T>::Contiguous2DArray(Contiguous2DArray<T>&& other) : rows_(other.rows_), cols_(other.cols_), arr_(other.arr_) {
	other.rows_ = 0;
	other.cols_ = 0;
	other.arr_ = nullptr;
}

template <typename T>
Contiguous2DArray<T>::~Contiguous2DArray() {
	destroyArray();
}

template <typename T>
T* Contiguous2DArray<T>::operator[](const size_t& row) const {
	if (arr_ == nullptr) {
		return nullptr;
	}
	return arr_[row];
}

template <typename T>
Contiguous2DArray<T>& Contiguous2DArray<T>::operator=(Contiguous2DArray<T>&& other) {
	if (this == &other) {
		return *this;
	}
	destroyArray();
	rows_ = other.rows_;
	cols_ = other.cols_;
	arr_ = other.arr_;
	other.rows_ = 0;
	other.cols_ = 0;
	other.arr_ = nullptr;
	return *this;
}

template <typename T>
void Contiguous2DArray<T>::destroyArray() {
	if (arr_ != nullptr) {
		delete[] arr_[0];
		delete[] arr_;
	}
}

#endif
