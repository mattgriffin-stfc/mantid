// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"

#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
double CudaV3D::normalize() {
  const double ND(norm3d(m_pt[0], m_pt[1], m_pt[2]));

  if (CudaMathExtensions::isZero(ND)) {
    raise("Unable to normalize a zero length vector.");
  }
  this->operator/=(ND);
  return ND;
}

__device__
double CudaV3D::distance(const CudaV3D &v) const noexcept {
  const CudaV3D &temp = (*this - v);

  return norm3d(temp.X(), temp.Y(), temp.Z());
}
__device__
bool CudaV3D::unitVector() const noexcept {
  const auto l = norm3d(m_pt[0], m_pt[1], m_pt[2]);
  return CudaMathExtensions::isZero(l - 1.);
}

}
}
