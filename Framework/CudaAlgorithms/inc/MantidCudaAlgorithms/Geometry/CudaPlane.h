// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidCudaAlgorithms/Geometry/CudaQuadratic.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"

namespace Mantid {
namespace CudaAlgorithms {

/**
 * The CudaPlane class (cut down, CUDA equivalent for Geometry::Plane)
 *
 * Constructable on device, usable on device.
 *
 * Defines a plane as a vector (unit) a distance.
 */
class CudaPlane : public CudaQuadratic {
public:
  /**
   * @brief CudaPlane
   * @param normV
   * @param dist
   */
  __device__
  CudaPlane(const CudaV3D &normV, const double dist);

  __device__
  void acceptVisitor(CudaLineIntersectVisit &A) const override;

  __device__
  int side(const CudaV3D &) const override;

  __device__
  void setBaseEqn() override;

  /// Distance from origin
  __inline__ __device__
  double getDistance() const { return Dist; }

  /// Normal to plane (+ve surface)
  __inline__ __device__
  const CudaV3D &getNormal() const { return NormV; }

private:
  /// norm vector of the plane
  const CudaV3D NormV;
  /// distance of the plane
  const double Dist;
};

}
}
