// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaCone.h"

#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"
#include "MantidCudaAlgorithms/CudaMathExtensions.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaCone::CudaCone(const CudaV3D &centre, const CudaV3D &norm,
                   const double cangle)
    : CudaQuadratic(), Centre(centre), Normal(norm), cangle(cangle) {
  CudaCone::setBaseEqn();
}

__device__
void CudaCone::acceptVisitor(CudaLineIntersectVisit &A) const {
  A.Accept(*this);
}

__device__
void CudaCone::setBaseEqn()
{
  const double c2(cangle * cangle);
  const double CdotN(Centre.scalar_prod(Normal));
  BaseEqn[0] = fma(-Normal[0], Normal[0], c2);                      // A x^2
  BaseEqn[1] = fma(-Normal[1], Normal[1], c2);                      // B y^2
  BaseEqn[2] = fma(-Normal[2], Normal[2], c2);                      // C z^2
  BaseEqn[3] = -2 * Normal[0] * Normal[1];                          // D xy
  BaseEqn[4] = -2 * Normal[0] * Normal[2];                          // E xz
  BaseEqn[5] = -2 * Normal[1] * Normal[2];                          // F yz
  BaseEqn[6] = 2.0 * fma(Normal[0], CdotN, -Centre[0] * c2);        // G x
  BaseEqn[7] = 2.0 * fma(Normal[1], CdotN, -Centre[1] * c2);        // H y
  BaseEqn[8] = 2.0 * fma(Normal[2], CdotN, -Centre[2] * c2);        // J z
  BaseEqn[9] = fma(c2, Centre.scalar_prod(Centre), -CdotN * CdotN); // K const
}

__device__
int CudaCone::side(const CudaV3D &Pt) const
{
  const CudaV3D &cR = Pt - Centre;
  double rptAngle = cR.scalar_prod(Normal);
  rptAngle *= rptAngle / cR.scalar_prod(cR);
  const double eqn = sqrt(rptAngle);

  return CudaMathExtensions::side(eqn - cangle);
}

}
}
