// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

namespace Mantid {
namespace CudaAlgorithms {
namespace CudaReduce {

/**
 * Reduce the values held in all lanes within the same warp
 * @param val of the current lane in the warp
 * @return the reduced (or partially reduced) value (lane0 holds full reduction)
 */
template<typename T>
__inline__ __device__
T warpReduce(T val) {
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

}
}
}
