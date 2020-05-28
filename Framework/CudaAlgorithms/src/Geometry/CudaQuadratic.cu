// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaQuadratic.h"

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaQuadratic::CudaQuadratic()
    : CudaSurface() {}

__device__
void CudaQuadratic::acceptVisitor(CudaLineIntersectVisit &A) const {
    A.Accept(*this);
}

__device__
double CudaQuadratic::eqnValue(const CudaV3D &Pt) const {
  double res(0.0);
  res += BaseEqn[0] * Pt.X() * Pt.X();
  res += BaseEqn[1] * Pt.Y() * Pt.Y();
  res += BaseEqn[2] * Pt.Z() * Pt.Z();
  res += BaseEqn[3] * Pt.X() * Pt.Y();
  res += BaseEqn[4] * Pt.X() * Pt.Z();
  res += BaseEqn[5] * Pt.Y() * Pt.Z();
  res += BaseEqn[6] * Pt.X();
  res += BaseEqn[7] * Pt.Y();
  res += BaseEqn[8] * Pt.Z();
  res += BaseEqn[9];
  return res;
}

__device__
int CudaQuadratic::side(const CudaV3D &Pt) const {
  double res = eqnValue(Pt);
  return CudaMathExtensions::side(res);
}

}
}
