// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"


namespace Mantid {
namespace CudaAlgorithms {

/**
 * The CudaBoundingBox class (cut down, CUDA equivalent for
 * Geometry::BoundingBox)
 *
 * Constructable on host + device, usable on device.
 *
 * A simple structure that defines an axis-aligned cuboid shaped bounding box
 * for a geometrical object. It is a thin structure containing the 6 points
 * that define the corners of the cuboid.
 */
class CudaBoundingBox {
public:
  /**
   * Constructor taking six points. If inconsistent points are defined, i.e.
   * xmin > xmax, then an error is thrown
   * @param xmax :: Value of maximum in X. It must be greater than xmin.
   * @param ymax :: Value of maximum in Y. It must be greater than ymin.
   * @param zmax :: Value of maximum in Z. It must be greater than zmin.
   * @param xmin :: Value of minimum in X. It must be less than xmax.
   * @param ymin :: Value of minimum in Y. It must be less than ymax.
   * @param zmin :: Value of minimum in Z. It must be less than zmax.
   */
  __inline__ __host__ __device__
  CudaBoundingBox(const double xmax, const double ymax, const double zmax,
                  const double xmin, const double ymin, const double zmin)
      : m_minPoint(xmin, ymin, zmin), m_maxPoint(xmax, ymax, zmax) {
    // Sanity check
    checkValid(xmax, ymax, zmax, xmin, ymin, zmin);
  }  

  /**
   * Do the given arguments form a valid bounding box, throws std::invalid
   * argument if not
   * @param xmax :: Value of maximum in X. It must be greater than xmin.
   * @param ymax :: Value of maximum in Y. It must be greater than ymin.
   * @param zmax :: Value of maximum in Z. It must be greater than zmin.
   * @param xmin :: Value of minimum in X. It must be less than xmax.
   * @param ymin :: Value of minimum in Y. It must be less than ymax.
   * @param zmin :: Value of minimum in Z. It must be less than zmax.
   */
  __inline__ __host__ __device__
  static void checkValid(const double xmax, const double ymax,
                         const double zmax, const double xmin,
                         const double ymin, const double zmin) {
    if (xmax < xmin || ymax < ymin || zmax < zmin) {
      #ifdef __CUDA_ARCH__
      // device implementation (cannot handle exceptions/use std)
      raise("Error creating bounding box, inconsistent values given");
      #else
      // host implementation
      std::ostringstream error;
      error << "Error creating bounding box, inconsistent values given:\n"
            << "\txmin=" << xmin << ", xmax=" << xmax << "\n"
            << "\tymin=" << ymin << ", ymax=" << ymax << "\n"
            << "\tzmin=" << zmin << ", zmax=" << zmax << "\n";
      throw std::invalid_argument(error.str());
      #endif
    }
  }

  /** @name Point access */
  /// Return the minimum value of X
  __inline__ __device__
  double xMin() const { return m_minPoint.X(); }
  /// Return the maximum value of X
  __inline__ __device__
  double xMax() const { return m_maxPoint.X(); }
  /// Return the minimum value of Y
  __inline__ __device__
  double yMin() const { return m_minPoint.Y(); }
  /// Return the maximum value of Y
  __inline__ __device__
  double yMax() const { return m_maxPoint.Y(); }
  /// Return the minimum value of Z
  __inline__ __device__
  double zMin() const { return m_minPoint.Z(); }
  /// Return the maximum value of Z
  __inline__ __device__
  double zMax() const { return m_maxPoint.Z(); }
  //// Returns the min point of the box
  __inline__ __device__
  const CudaV3D &minPoint() const { return m_minPoint; }
  /// Returns the min point of the box
  __inline__ __device__
  const CudaV3D &maxPoint() const { return m_maxPoint; }

  /**
   * Query whether the given point is inside the bounding box within a tolerance
   * defined by Mantid::Geometry::Tolerance.
   * @param point :: The point to query
   * @returns True if the point is within the bounding box, false otherwise
   */
  __device__
  bool isPointInside(const CudaV3D &point) const;

  /**
   * Generate a random point within this box assuming the 3 numbers given
   * are random numbers in the range (0,1) & selected from a flat distribution.
   * @param r1 Flat random number in range (0,1)
   * @param r2 Flat random number in range (0,1)
   * @param r3 Flat random number in range (0,1)
   * @return A new point within the box such that isPointInside(pt) == true
   */
  __device__
  CudaV3D generatePointInside(const double r1,
                              const double r2,
                              const double r3) const;

private:
  /// The minimum point of the axis-aligned box
  CudaV3D m_minPoint;
  /// The maximum point of the axis-aligned box
  CudaV3D m_maxPoint;
};

}
}
