// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaCylinder.h"

#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"
#include "MantidCudaAlgorithms/CudaMathExtensions.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaCylinder::CudaCylinder(const CudaV3D &centre, const CudaV3D &norm,
                           const double radius)
    : CudaQuadratic(), Centre(centre), Normal(norm), Radius(radius), Nvec(0) {
  CudaCylinder::setBaseEqn();

  for (int i = 0; i < 3; i++) {
    if (fabs(Normal[i]) > (1.0 - CudaMathExtensions::CudaTolerance)) {
      Nvec = i + 1;
      return;
    }
  }
}

__device__
void CudaCylinder::acceptVisitor(CudaLineIntersectVisit &A) const {
  A.Accept(*this);
}

__device__
void CudaCylinder::setBaseEqn() {
  const double CdotN = Centre.scalar_prod(Normal);
  BaseEqn[0] = fma(-Normal.X(), Normal.X(), 1.0);         // A x^2
  BaseEqn[1] = fma(-Normal.Y(), Normal.Y(), 1.0);         // B y^2
  BaseEqn[2] = fma(-Normal.Z(), Normal.Z(), 1.0);         // C z^2
  BaseEqn[3] = -2.0 * Normal.X() * Normal.Y();            // D xy
  BaseEqn[4] = -2.0 * Normal.X() * Normal.Z();            // E xz
  BaseEqn[5] = -2.0 * Normal.Y() * Normal.Z();            // F yz
  BaseEqn[6] = 2.0 * fma(Normal.X(), CdotN, -Centre.X()); // G x
  BaseEqn[7] = 2.0 * fma(Normal.Y(), CdotN, -Centre.Y()); // H y
  BaseEqn[8] = 2.0 * fma(Normal.Z(), CdotN, -Centre.Z()); // J z
  BaseEqn[9] = fma(-Radius, Radius, fma(-CdotN, CdotN,
                                        Centre.scalar_prod(Centre))); // K const
}

__device__
int CudaCylinder::side(const CudaV3D &Pt) const {
  if (Nvec) // Nvec =1-3 (point to exclude == Nvec-1)
  {
    if (Radius > 0.0) {
      double x = Pt[Nvec % 3] - Centre[Nvec % 3];
      x *= x;
      double y = Pt[(Nvec + 1) % 3] - Centre[(Nvec + 1) % 3];

      y *= y;
      const double displace = x + y - Radius * Radius;
      return !CudaMathExtensions::isZero(displace / Radius) *
              ((displace > 0.0) * 2 - 1);
    } else {
      return -1;
    }
  }

  return CudaQuadratic::side(Pt);
}

}
}
