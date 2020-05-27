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
class CudaV3D;
class CudaSurface;

/**
 * CudaRule, trimmed down CUDA equivalent for Geometry::Rule
 *
 * Base class for all CudaRules
 */
class CudaRule {
public:
  /**
   * @return whether the point is within the object
   */
  __device__
  virtual bool isValid(const CudaV3D &) const = 0;
};


/**
 * CudaIntersection, trimmed down CUDA equivalent to Geometry::Intersection
 *
 * Combines two Rule objects in an intersection (A AND B)
 */
class CudaIntersection : public CudaRule {
private:
  const CudaRule * A;
  const CudaRule * B;

public:
  __inline__ __device__
  CudaIntersection(const CudaRule * left, const CudaRule * right)
      : CudaRule(), A(left), B(right) {}

  __device__
  bool isValid(const CudaV3D &) const override;
};

/**
 * CudaUnion, trimmed down CUDA equivalent to Geometry::Union
 *
 * Combines two Rule objects in an intersection (A OR B)
 */
class CudaUnion : public CudaRule {
private:
  const CudaRule * A;
  const CudaRule * B;

public:
  __inline__ __device__
  CudaUnion(const CudaRule * left, const CudaRule * right)
      : CudaRule(), A(left), B(right) {}

  __device__
  bool isValid(const CudaV3D &) const override;
};

/**
 * CudaSurfPoint, trimmed down CUDA equivalent to Geometry::CudaSurfPoint
 *
 * Acts as an interface between a CudaSurface object and a rule.
 */
class CudaSurfPoint : public CudaRule {
private:
  const CudaSurface * m_key;
  const int sign;

public:
  __inline__ __device__
  CudaSurfPoint(const CudaSurface * key, const int sign)
      : CudaRule(), m_key(key), sign(sign) {}

  __device__
  bool isValid(const CudaV3D &) const override;
};

}
}
