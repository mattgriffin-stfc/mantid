// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaPlane.h"

#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"


namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaPlane::CudaPlane(const CudaV3D &norm, const double distance)
    : CudaQuadratic(), NormV(norm), Dist(distance) {

  CudaPlane::setBaseEqn();
}

__device__
void CudaPlane::acceptVisitor(CudaLineIntersectVisit &A) const {
    A.Accept(*this);
}


__device__
int CudaPlane::side(const CudaV3D &A) const {
  const double Dp = NormV.scalar_prod(A) - Dist;

  return CudaMathExtensions::side(Dp);
}

__device__
void CudaPlane::setBaseEqn() {
  BaseEqn[0] = 0.0;      // A x^2
  BaseEqn[1] = 0.0;      // B y^2
  BaseEqn[2] = 0.0;      // C z^2
  BaseEqn[3] = 0.0;      // D xy
  BaseEqn[4] = 0.0;      // E xz
  BaseEqn[5] = 0.0;      // F yz
  BaseEqn[6] = NormV.X(); // G x
  BaseEqn[7] = NormV.Y(); // H y
  BaseEqn[8] = NormV.Z(); // J z
  BaseEqn[9] = -Dist;    // K const
}

}
}
