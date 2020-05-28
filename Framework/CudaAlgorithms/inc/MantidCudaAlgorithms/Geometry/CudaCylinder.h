// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

#include "MantidCudaAlgorithms/Geometry/CudaQuadratic.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"

namespace Mantid {
namespace CudaAlgorithms {

class CudaLineIntersectVisit;

/**
 * The CudaCylinder class (cut down, CUDA equivalent for Geometry::Cylinder)
 *
 * Constructable on device, usable on device.
 *
 * Defines a cylinder as a centre point (on main axis) a vector from that point
 * (unit) and a radius.
 */
class CudaCylinder : public CudaQuadratic {
public:
  /**
   * Constructor for CudaCylinder
   * @param centre of the cylinder
   * @param norm vector of the cylinder
   * @param radius of the cylinder
   */
  __device__
  CudaCylinder(const CudaV3D &centre, const CudaV3D &norm,
               const double radius);

  __device__
  void setBaseEqn() override;

  __device__
  int side(const CudaV3D &) const override;

  __device__
  void acceptVisitor(CudaLineIntersectVisit &A) const override;

  /// Return centre point
  __inline__ __device__
  const CudaV3D &getCentre() const { return Centre; }

  /// Return Central line
  __inline__ __device__
  const CudaV3D &getNormal() const { return Normal; }

  /// Return the Radius
  __inline__ __device__
  double getRadius() const { return Radius; }

private:
  /// centre point of the cylinder
  const CudaV3D Centre;
  /// the cylinders normal
  const CudaV3D Normal;
  /// the radius of the cylinder
  const double Radius;
  /// the axis aligned normal
  unsigned int Nvec;
};

}
}
