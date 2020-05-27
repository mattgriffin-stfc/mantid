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
 * The CudaCone class (cut down, CUDA equivalent for Geometry::Cone)
 *
 * Constructable on device, usable on device.
 *
 * Defines a cone as a centre point (on main axis) a vector from that point
 * (unit) a radius and an angle of incidence (cosine angle).
 */
class CudaCone : public CudaQuadratic {
public:
  /**
   * Constructor for CudaCone
   * @param centre of the bottom base of the cone
   * @param norm vector of the cone
   * @param cangle of the cone's incidence
   */
  __device__
  CudaCone(const CudaV3D &centre, const CudaV3D &norm, const double cangle);

  __device__
  void setBaseEqn() override;

  __device__
  int side(const CudaV3D &pt) const override;

  __device__
  void acceptVisitor(CudaLineIntersectVisit &visit) const override;

  /// Return centre point
  __inline__ __device__
  const CudaV3D &getCentre() const { return Centre; }
  /// Return Central line
  __inline__ __device__
  const CudaV3D &getNormal() const { return Normal; }
  /// Edge angle
  __inline__ __device__
  double getCosAngle() const { return cangle; }

private:
  /// V3D for centre
  CudaV3D Centre;
  /// Normal
  CudaV3D Normal;
  /// Cos(angle)
  double cangle;
};

}
}
