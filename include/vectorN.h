#pragma once
#include <memory>

template <class T>
class vectorN
{
public:
  vectorN() {
    _nCols = 0;
    _buffer = nullptr;
  }

  vectorN(int p_cols) {
    init(p_cols);
  }

  ~vectorN(void) {
    delete[] _buffer;
  }

  void init(int p_cols) {
    _nCols = p_cols;
    _buffer = new T[p_cols];    
  }

  T* getVector() {
    return _buffer;
  }

  void setVector(T* p_buffer) {
    memcpy(_buffer, p_buffer, sizeof(T) * _nCols);
  }

  void setVector(vectorN<T>* p_vector) {
    setVector(p_vector->getVector());
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

  vectorN<T>& operator=(const vectorN<T> &v )
  {
      _nCols = v._nCols;
      setVector(v._buffer);
      return *this;
  }

  vectorN<T>& operator=(const vectorN<T> *v )
  {
      _nCols = v->_nCols;
      _buffer = v->_buffer;
      return *this;
  }

  vectorN<T>* operator+(const vectorN<T>& v) const
  {
    vectorN<T> *result = new vectorN<T>(v.size());
    result->set(0);

    for(auto j = 0; j < _nCols; j++) {
      result->_buffer[j] = v._buffer[j] + _buffer[j];
    }

    return result;
  }

  bool operator==(const vectorN<T>& v) const
  {
    bool result = true;
    for(auto j = 0; j < _nCols; j++) {
      result = result && _buffer[j] == v._buffer[j];
    }        
    return result;
  }

  vectorN<T>* operator*(T& c)
  {
    vectorN<T>* result = new vectorN<double>(_nCols);
    result->set(0);

    for(auto j = 0; j < _nCols; j++) {
      result->_buffer[j] = c * _buffer[j];
    }

    return result;
  }

  void operator+= (const vectorN<T>& v) const
  {
    for(auto j = 0; j < _nCols; j++) {
      _buffer[j] += v._buffer[j];
    }
  }

  T &operator[](int i) const
  {
      if( i > _nCols )
      {
        //return ;
      }
      return _buffer[i];
  }


private:
  int _nCols;
  
  T* _buffer;
};