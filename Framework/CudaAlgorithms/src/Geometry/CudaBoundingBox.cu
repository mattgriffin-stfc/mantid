// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaBoundingBox.h"

#include "MantidCudaAlgorithms/CudaMathExtensions.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
bool CudaBoundingBox::isPointInside(const CudaV3D &point) const {
  return (point.X() <= xMax() + CudaMathExtensions::CudaTolerance) &
         (point.X() >= xMin() - CudaMathExtensions::CudaTolerance) &
         (point.Y() <= yMax() + CudaMathExtensions::CudaTolerance) &
         (point.Y() >= yMin() - CudaMathExtensions::CudaTolerance) &
         (point.Z() <= zMax() + CudaMathExtensions::CudaTolerance) &
         (point.Z() >= zMin() - CudaMathExtensions::CudaTolerance);
}

__device__
CudaV3D CudaBoundingBox::generatePointInside(const double r1, const double r2,
                                             const double r3) const {
  return CudaV3D(fma(r1, xMax() - xMin(), xMin()),
                 fma(r2, yMax() - yMin(), yMin()),
                 fma(r3, zMax() - zMin(), zMin()));
}

}
}
