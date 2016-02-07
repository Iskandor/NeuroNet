#pragma once

template <class T>
class matrix2
{
public:
  matrix2(int p_rows, int p_cols) {
    _nRows = p_rows;
    _nCols = p_cols;
    _buffer = new T[p_rows * p_cols];
  }

  ~matrix2(void) {
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

  void set(T p_value)
  {
    for(auto i = 0; i < _nRows * _nCols; i++)
    {
      _buffer[i] = p_value;
    }
  }

  int size() const
  {
    return _nRows * _nCols;
  }

private:
  int _nRows;
  int _nCols;  
  
  T* _buffer;
};

