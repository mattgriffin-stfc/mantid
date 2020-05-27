// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidCudaAlgorithms/Geometry/CudaSurface.h"

namespace Mantid {
namespace CudaAlgorithms {

/**
 * The CudaQuadratic abstract class (cut down, CUDA equivalent for
 * Geometry::Quadratic)
 *
 * Constructable on device, usable on device.
 *
 * Defines a surface as a quadratic equation of ten terms.
 */
class CudaQuadratic : public CudaSurface {
public:
  /**
   * Constructor for CudaQuadratic
   */
  __device__
  CudaQuadratic();

  __device__
  void acceptVisitor(CudaLineIntersectVisit &A) const override;

  __device__
  int side(const CudaV3D &) const override;

  /**
   * Abstract set baseEqn that any Quadtratic surface should define
   */
  __device__
  virtual void setBaseEqn() = 0;

  /**
   * Helper function to calcuate the value of the equation at a fixed point
   * @param Pt :: Point to determine the equation surface
   * value at
   * @return value Eqn(Pt) : -ve inside +ve outside
   */
  __device__
  double eqnValue(const CudaV3D &) const;

  /// return the base equation
  __inline__ __device__
  const double * copyBaseEqn() const {
    return BaseEqn;
  }

protected:
  /// Quadratic equation of the surface
  double BaseEqn[10];
};

}
}
