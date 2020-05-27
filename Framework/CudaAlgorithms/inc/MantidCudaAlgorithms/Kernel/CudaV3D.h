// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidKernel/V3D.h"

#include <cuda_runtime.h>

#include <math.h>

namespace Mantid {
namespace CudaAlgorithms {

/**
 * CudaV3D, CUDA equivalent of Kernel::V3D
 *
 * Provides methods for interacting with a 3D vector.
 */
class CudaV3D final {
public:
  __host__
  CudaV3D(const Kernel::V3D & v3d) noexcept {
    m_pt[0] = v3d.X();
    m_pt[1] = v3d.Y();
    m_pt[2] = v3d.Z();
  }

  __device__
  CudaV3D(const double x, const double y, const double z) noexcept {
    m_pt[0] = x;
    m_pt[1] = y;
    m_pt[2] = z;
  }

  __host__ __device__
  CudaV3D() noexcept {
    m_pt[0] = 0.0;
    m_pt[1] = 0.0;
    m_pt[2] = 0.0;
  }

  /**
   * Addtion operator
   * @param v :: Vector to add
   * @return *this+v;
   */
  __inline__ __device__
  CudaV3D operator+(const CudaV3D &v) const noexcept {
    return CudaV3D(m_pt[0] + v.m_pt[0], m_pt[1] + v.m_pt[1], m_pt[2]
            + v.m_pt[2]);
  }

  /**
   * Subtraction operator
   * @param v :: Vector to sub.
   * @return *this-v;
   */
  __inline__ __device__
  CudaV3D operator-(const CudaV3D &v) const noexcept {
    return CudaV3D(m_pt[0] - v.m_pt[0], m_pt[1] - v.m_pt[1], m_pt[2]
            - v.m_pt[2]);
  }

  /**
    Self-Addition operator
    @param v :: Vector to add.
    @return *this+=v;
  */
  __inline__ __device__
  CudaV3D &operator+=(const CudaV3D &v) noexcept {
    m_pt[0] += v.m_pt[0];
    m_pt[1] += v.m_pt[1];
    m_pt[2] += v.m_pt[2];

    return *this;
  }

  /**
    Scalar product
    @param D :: value to scale
    @return this * D
   */
  __inline__ __device__
  CudaV3D operator*(const double D) const noexcept {
    return CudaV3D(m_pt[0] * D, m_pt[1] * D, m_pt[2] * D);
  }

  /**
    Scalar product
    @param D :: value to scale
    @return this *= D
  */
  __inline__ __device__
  CudaV3D &operator*=(const double D) noexcept {
    m_pt[0] *= D;
    m_pt[1] *= D;
    m_pt[2] *= D;

    return *this;
  }

  /**
    Scalar division
    @param D :: value to scale
    @return this /= D
  */
  __inline__ __device__
  CudaV3D &operator/=(const double D) noexcept {
    m_pt[0] /= D;
    m_pt[1] /= D;
    m_pt[2] /= D;
    return *this;
  }

  /**
    Set is x position
    @param xx :: The X coordinate
  */
  __inline__ __device__
  void setX(const double xx) noexcept { m_pt[0] = xx; }

  /**
    Set is y position
    @param yy :: The Y coordinate
  */
  __inline__ __device__
  void setY(const double yy) noexcept { m_pt[1] = yy; }

  /**
    Set is z position
    @param zz :: The Z coordinate
  */
  __inline__ __device__
  void setZ(const double zz) noexcept { m_pt[2] = zz; }

  __inline__ __device__ __host__
  constexpr double X() const noexcept { return m_pt[0]; }
  __inline__ __device__ __host__
  constexpr double Y() const noexcept { return m_pt[1]; }
  __inline__ __device__ __host__
  constexpr double Z() const noexcept { return m_pt[2]; }

  /**
   * Returns the axis value based in the index provided
   * @param index :: 0=x, 1=y, 2=z
   * @return a double value of the requested axis
   */
  __inline__ __device__
  constexpr double operator[](const size_t index) const noexcept {
    return m_pt[index];
  }

  /**
   * Returns the axis value based in the index provided
   * @param index :: 0=x, 1=y, 2=z
   * @return a double value of the requested axis
   */
  __inline__ __device__
  double &operator[](const size_t index) noexcept {
    return m_pt[index];
  }

  /**
   * Normalises the vector and returns its original length
   * @return the norm of the vector before normalization
   */
  __device__
  double normalize();

  /**
   * Calculates the scalar cross product. Returns (this * v).
   * @param v :: The second vector to include in the calculation
   * @return The cross product of the two vectors (this * v)
   */
  __inline__ __device__
  double scalar_prod(const CudaV3D &v) const noexcept {
    return fma(m_pt[0],  v.m_pt[0],
           fma(m_pt[1],  v.m_pt[1],
               m_pt[2] * v.m_pt[2]));
  }

  /**
   * Calculates the vector cross product. Returns (this * v).
   * @param v :: The second vector to include in the calculation
   * @return The cross product of the two vectors (this * v)
   */
  __inline__ __device__
  CudaV3D cross_prod(const CudaV3D &v) const noexcept {
    return CudaV3D(fma(m_pt[1], v.m_pt[2], -m_pt[2] * v.m_pt[1]),
                   fma(m_pt[2], v.m_pt[0], -m_pt[0] * v.m_pt[2]),
                   fma(m_pt[0], v.m_pt[1], -m_pt[1] * v.m_pt[0]));
  }
  /**
   * Calculates the distance between two vectors
   * @param v :: The second vector to include in the calculation
   * @return The distance between the two vectors
   */
  __device__
  double distance(const CudaV3D &v) const noexcept;

  /**
   * @return whether the vector meets the criteria of a unit vector
   */
  __device__
  bool unitVector() const noexcept;

private:
  double m_pt[3];
};


}
}
