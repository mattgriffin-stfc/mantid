// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

namespace thrust {
template<typename T>
class complex;
}

namespace Mantid {
namespace CudaAlgorithms {

/**
 * CudaMathExtensions, extension functions to the standard CUDA math library,
 * specifically optimized for CUDA.
 *
 * Functions must remain synchonous within a warp.
 */
namespace CudaMathExtensions {

__device__
__constant__
const double CudaTolerance = 1.0e-06;

/**
 * "clamps" a value between two values. Equivalent to std::clamp
 * @param v the value to clamp
 * @param lo  the min value
 * @param hi  the maximum value
 * @return either, v, lo or hi
 */
__inline__ __device__
double fclamp(const double v, const double lo, const double hi) {
  return fmin(fmax(v, lo), hi);
}

/**
 * Check if value is equal to zero within a given tolerance
 * @param v the value to check
 * @return whether v is equal to 0 with tolerance
 */
__inline__ __device__
bool isZero(const double v) {
  return fabs(v) < CudaTolerance;
}

/**
 * side returns an int representing whether the val is greater (1), less than
 * (-1) or equal to zero (0)
 * @param val to test
 * @return 1 if > 0, -1 < 0 else 0
 */
__inline__ __device__
int side(const double v) {
  return !isZero(v) * (((v > 0) << 1) - 1);
}

/**
 *  Solves Complex Quadratic
 * @param coef  x, y and z coefficients in the order
 *        \f[ Ax^2+Bx+C \f].
 * @param d1  first complex root of the equation
 * @param d1  second complex root of the equation
 * @return number of unique solutions
 */
__device__
int solveQuadratic(const double3 &coef,
                   thrust::complex<double> &d1,
                   thrust::complex<double> &d2);

}
}
}
