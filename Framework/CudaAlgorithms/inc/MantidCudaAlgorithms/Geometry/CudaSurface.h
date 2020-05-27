// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

namespace Mantid {
namespace CudaAlgorithms {
class CudaV3D;
class CudaLineIntersectVisit;

/**
 * CudaSurface, trimmed down CUDA equivalent of the Geometry::Surface interface.
 *
 * Interface for a geometric surface.
 */
class CudaSurface {
public:

  /**
   * Accept a tracked visit to the surface
   * @param A the line intersect visit that interacts with the surface
   */
  __device__
  virtual void acceptVisitor(CudaLineIntersectVisit &A) const = 0;

  /**
   * Check which side of the surface a point falls on.
   * @param pt the 3D point to check
   * @return 0 if on surface, 1 if true to surface, -1 if false to surface
   */
  __device__
  virtual int side(const CudaV3D &pt) const = 0;
};

} 
} 
