#pragma once

template <class T>
class matrix3
{
public:
  matrix3(int p_x, int p_y, int p_z) {
    _nx = p_x;
    _ny = p_y;
    _nz = p_z;
    _buffer = new T[p_x * p_y * p_z];
  }

  ~matrix3(void) {
    delete[] _buffer;
  }

  T* getMatrix() {
    return _buffer;
  }

  void setMatrix(T* p_buffer) {
    _buffer = p_buffer;
  }

  T at(int p_i, int p_j, int p_k) {
    return _buffer[p_i * _nx * _ny + p_j * _nx + p_k];
  }

  void set(int p_i, int p_j, int p_k, T p_value) {
    _buffer[p_i * _nx * _ny + p_j * _nx + p_k] = p_value;
  }

  void set(T p_value)
  {
    for(auto i = 0; i < _nx * _ny * _nz; i++)
    {
      _buffer[i] = p_value;
    }
  }

  int size() const
  {
    return _nx * _ny * _nz;
  }

private:
  int _nx;
  int _ny;
  int _nz;
  
  T* _buffer;
};