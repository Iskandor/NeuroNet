#pragma once

template <class T>
class vectorN
{
public:
  vectorN(int p_cols) {
    _nCols = p_cols;
    _buffer = new T[p_cols];
  }

  ~vectorN(void) {
    delete[] _buffer;
  }

  T* getVector() {
    return _buffer;
  }

  void setVector(T* p_buffer) {
    memcpy(_buffer, p_buffer, sizeof(T) * _nCols);
  }

  T at(int p_i) {
    return _buffer[p_i];
  }

  void set(int p_i, T p_value) {
    _buffer[p_i] = p_value;
  }

  void set(T p_value)
  {
    for(auto i = 0; i < _nCols; i++)
    {
      _buffer[i] = p_value;
    }
  }

  int size() const
  {
    return _nCols;
  }

private:
  int _nCols;  
  
  T* _buffer;
};