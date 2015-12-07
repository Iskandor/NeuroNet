#pragma once

template <class T>
class matrix
{
public:
  matrix(int p_rows, int p_cols) {
    _nRows = p_rows;
    _nCols = p_cols;
    _buffer = new T[p_rows * p_cols];
    memset(_buffer, 0, sizeof(T) * _nCols * _nRows);
  }

  ~matrix(void) {
    delete[] _buffer;
  }

  T* getMatrix() {
    return _buffer;
  }

  void setMatrix(T* p_buffer) {
    _buffer = p_buffer;
  }

  T at(int p_i, int p_j) {
    return _buffer[p_i * _nCols + p_j];
  }

  void set(int p_i, int p_j, T p_value) {
    _buffer[p_i * _nCols + p_j] = p_value;
  }

private:
  int _nRows;
  int _nCols;  
  
  T* _buffer;
};

