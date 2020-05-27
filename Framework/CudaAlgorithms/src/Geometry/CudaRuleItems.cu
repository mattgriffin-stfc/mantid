// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaRules.h"

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/Geometry/CudaSurface.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
bool CudaIntersection::isValid(const CudaV3D &Pt) const
{
  return A->isValid(Pt) & B->isValid(Pt);
}

__device__
bool CudaUnion::isValid(const CudaV3D &Pt) const
{
  return A->isValid(Pt) | B->isValid(Pt);
}

__device__
bool CudaSurfPoint::isValid(const CudaV3D &Pt) const
{
  return (m_key->side(Pt) * sign) >= 0;
}

}
}
