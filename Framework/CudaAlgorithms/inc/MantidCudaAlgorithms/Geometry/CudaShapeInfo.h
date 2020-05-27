// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

#include "MantidGeometry/Rendering/ShapeInfo.h"


namespace Mantid {

using namespace Geometry::detail;

namespace CudaAlgorithms {
class CudaV3D;

/**
 * CudaShapeInfo, cut down CUDA equivalent to Geometry::detail::ShapeInfo.
 *
 * Holds information about the sample shape definition to optimize geometric
 * operations.
 */
class CudaShapeInfo {
public:
  struct CuboidGeometry {
    const CudaV3D &leftFrontBottom;
    const CudaV3D &leftFrontTop;
    const CudaV3D &leftBackBottom;
    const CudaV3D &rightFrontBottom;
  };
  struct SphereGeometry {
    const CudaV3D &centre;
    double radius;
  };
  struct CylinderGeometry {
    const CudaV3D &centreOfBottomBase;
    const CudaV3D &xTransform;
    const CudaV3D &yTransform;
    const CudaV3D &zTransform;
    double radius;
    double height;
  };
  struct HollowCylinderGeometry {
    const CudaV3D &centreOfBottomBase;
    const CudaV3D &xTransform;
    const CudaV3D &yTransform;
    const CudaV3D &zTransform;
    double innerRadius;
    double radius;
    double height;
  };

  __host__
  CudaShapeInfo(const double height, const double radius,
                const double innerRadius, const ShapeInfo::GeometryShape shape,
                const CudaV3D * points);

  __device__
  CylinderGeometry cylinderGeometry() const;
  __device__
  CuboidGeometry cuboidGeometry() const;
  __device__
  HollowCylinderGeometry hollowCylinderGeometry() const;
  __device__
  SphereGeometry sphereGeometry() const;

  __inline__ __device__
  ShapeInfo::GeometryShape shape() const { return m_shape; }

private:
  /// height for cone, cylinder and hollow cylinder
  const double m_height;
  /// radius for the sphere, cone and cylinder also outer radius for hollow
  /// cylinder
  const double m_radius;
  /// Inner radius for hollow cylinder
  const double m_innerRadius;
  /// the approximate shape this object models
  const ShapeInfo::GeometryShape m_shape;
  /// the points of the shape
  const CudaV3D * m_points;
};

}
}
