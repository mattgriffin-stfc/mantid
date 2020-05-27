// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaSphere.h"

#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaSphere::CudaSphere(const CudaV3D &centre, const double radius)
    : CudaQuadratic(), m_centre(centre), m_radius(radius) {

  CudaSphere::setBaseEqn();
}

__device__
void CudaSphere::acceptVisitor(CudaLineIntersectVisit &A) const {
  A.Accept(*this);
}

__device__
void CudaSphere::setBaseEqn() {
  BaseEqn[0] = 1.0;                                                  // A x^2
  BaseEqn[1] = 1.0;                                                  // B y^2
  BaseEqn[2] = 1.0;                                                  // C z^2
  BaseEqn[3] = 0.0;                                                  // D xy
  BaseEqn[4] = 0.0;                                                  // E xz
  BaseEqn[5] = 0.0;                                                  // F yz
  BaseEqn[6] = -2.0 * m_centre[0];                                   // G x
  BaseEqn[7] = -2.0 * m_centre[1];                                   // H y
  BaseEqn[8] = -2.0 * m_centre[2];                                   // J z
  BaseEqn[9] = m_centre.scalar_prod(m_centre) - m_radius * m_radius; // K const
}

__device__
int CudaSphere::side(const CudaV3D &Pt) const {
  const double xdiff(Pt.X() - m_centre.X()),
          ydiff(Pt.Y() - m_centre.Y()),
          zdiff(Pt.Z() - m_centre.Z());

  const double displace =
          sqrt(fma(xdiff, xdiff, fma(ydiff, ydiff, zdiff * zdiff))) - m_radius;

  return CudaMathExtensions::side(displace);
}

}
}
