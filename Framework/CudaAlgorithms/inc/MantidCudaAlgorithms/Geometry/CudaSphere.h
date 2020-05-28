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
 * The CudaSphere class (cut down, CUDA equivalent for Geometry::Sphere)
 *
 * Constructable on device, usable on device.
 *
 * Defines a sphere as a centre point (on main axis) and a radius.
 */
class CudaSphere : public CudaQuadratic {
public:
  /**
   * Constructor for CudaSphere
   * @param centre of the sphere
   * @param radius of the sphere
   */
  __device__
  CudaSphere(const CudaV3D &centre, const double radius);

  __device__
  void acceptVisitor(CudaLineIntersectVisit &A) const override;

  __device__
  int side(const CudaV3D &) const override;

  __device__
  void setBaseEqn() override;

  /// return the centre point of the sphere
  __inline__ __device__
  const CudaV3D &getCentre() const { return m_centre; }

  /// return the sphere's radius
  __inline__ __device__
  double getRadius() const { return m_radius; }

private:
  /// Point for centre
  const CudaV3D m_centre;
  /// Radius of sphere
  const double m_radius;
};

}
}
