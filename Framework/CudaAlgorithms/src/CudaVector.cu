// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/CudaVector.h"

#include "MantidCudaAlgorithms/Geometry/CudaTrack.h"

namespace Mantid {
namespace CudaAlgorithms {

template <class T>
__device__
void CudaVector<T>::insert(const unsigned int p, T &value) {
  if (m_nValues >= m_nValuesMax) {
    raise("CudaVector: Vector too small, increase allocated memory.");
  } else if (p > m_nValues) {
    raise("CudaVector: Index exceeds current capacity.");
  }

  /* move all data at right side of the array */
  for(unsigned int i = m_nValues; i > p; i--) {
    m_values[i] = m_values[i - 1];
  }
  __syncthreads();

  /* insert value at the proper position */
  m_values[p] = value;
  ++m_nValues;
}

// forward declarations
template class CudaVector<CudaLink>;
template class CudaVector<CudaIntersectionPoint>;
}
}
