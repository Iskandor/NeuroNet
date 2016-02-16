#pragma once

template <class T>
class matrix2
{
public:
  enum TYPE {
      NONE = 0,
      ZERO = 1,
      UNITARY = 2
  };

  matrix2() {
    _nRows = 0;
    _nCols = 0;
    _buffer = nullptr;    
  }

  matrix2(int p_rows, int p_cols) {
    init(p_rows, p_cols);
  }

  void init(int p_rows, int p_cols, TYPE p_type = NONE) {
    _nRows = p_rows;
    _nCols = p_cols;
    _buffer = new T[p_rows * p_cols];

    switch(p_type) {
      case ZERO:
        set(0);
      break;
      case UNITARY:
        set(0);
        if (_nRows == _nCols) {
          for(int i = 0; i < _nRows; i++) {
            _buffer[i * _nRows + i] = 1;
          }
        }
      break;
    }
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

  vectorN<T>* operator*(vectorN<T>& v)
  {
    vectorN<T>* result = new vectorN<double>(_nRows);
    result->set(0);

    for(auto i = 0; i < _nRows; i++) {
      for(auto j = 0; j < _nCols; j++) {
        (*result)[i] += v[j] * _buffer[i * _nCols + j];
      }
    }

    return result;
  }

  void operator*= (const T& c) const
  {
    for(auto i = 0; i < _nRows; i++) {
      for(auto j = 0; j < _nCols; j++) {
        _buffer[i * _nCols + j] *= c;
      }
    }
  }

  void operator+= (const matrix2<T>& m) const
  {
    for(auto i = 0; i < _nRows; i++) {
      for(auto j = 0; j < _nCols; j++) {
        _buffer[i * _nCols + j] += m._buffer[i * _nCols + j];
      }
    }
  }

private:
  int _nRows;
  int _nCols;  
  
  T* _buffer;
};

