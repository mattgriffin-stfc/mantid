// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"

namespace Mantid {
namespace CudaAlgorithms {

/**
 * Naive implementation of std::vector for CUDA. Requires memory to be passed
 * to the implementation enabling use of local, shared, global or const memory.
 */
template <class T>
class CudaVector {
public:
  /**
   * Constructor for CudaVector
   * @param values memory for the container class
   * @param maxValues max values supported by the memory
   */
  __inline__ __device__
  CudaVector(T * values, const unsigned int maxValues)
      : m_values(values), m_nValues(0), m_nValuesMax(maxValues) {}

  /**
   * Add an element at the given index
   * @param i index to add the value at
   * @param value to add
   */
  __device__
  void insert(const unsigned int i, T &value);

  /**
   * Remove all values from memory
   */
  __inline__ __device__
  void clear() {
    m_nValues = 0;
  }

  /**
   * Add a value to the back oo the vector
   * @param value to add
   */
  __inline__ __device__
  void emplace_back(T &value) {
    if (m_nValues >= m_nValuesMax) {
      raise("CudaVector: Vector too large, reduce block size or increase "
            "allocated memory.");
    }

    m_values[m_nValues++] = value;
  }

  /**
   * Index based accessor
   * @param index to fetch the value from
   * @return the value at the given index
   */
  __inline__ __device__
  T& operator[](unsigned int index) {
    if (index >= m_nValues) {
      raise("CudaVector: Index exceeds current capacity.");
    }

    return m_values[index];
  }

  /**
   * Const index based accessor
   * @param index to fetch the value from
   * @return the value at the given index
   */
  __inline__ __device__
  const T& operator[](unsigned int index) const {
    if (index >= m_nValues) {
      raise("CudaVector: Index exceeds current capacity.");
    }

    return m_values[index];
  }

  /// return the size of the vector
  __inline__ __device__
  unsigned int size() const { return m_nValues; }

private:
  /// internal data storage for values
  T * m_values;
  /// number of values in the vector
  unsigned int m_nValues;
  /// max number of values in the vector
  const unsigned int m_nValuesMax;
};

}
}
