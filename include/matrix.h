#pragma once

template <class T>
class matrix
{
public:
  matrix(int p_rows, int p_cols) {
    _nRows = p_rows;
    _nCols = p_cols;
    _buffer = new T[p_cols*p_rows];
    memset(_buffer, 0, sizeof(T) * _nRows * _nCols);
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

  T at(int p_j, int p_i) {
    return _buffer[p_i * _nRows + p_j];
  }

  void set(int p_j, int p_i, T p_value) {
    _buffer[p_i * _nRows + p_j] = p_value;
  }

private:
  int _nRows;
  int _nCols;  
  
  T* _buffer;
};

