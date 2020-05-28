// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaShapeInfo.h"

#include "MantidKernel/Matrix.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"

namespace Mantid {
namespace CudaAlgorithms {

__host__
CudaShapeInfo::CudaShapeInfo(const double height, const double radius,
                             const double innerRadius,
                             const ShapeInfo::GeometryShape shape,
                             const CudaV3D * points)
    : m_height(height), m_radius(radius), m_innerRadius(innerRadius),
      m_shape(shape), m_points(points) {}

__device__
CudaShapeInfo::CuboidGeometry CudaShapeInfo::cuboidGeometry() const {
  if (m_shape != ShapeInfo::GeometryShape::CUBOID) {
    raise("Cannot generate cuboid geometry as shape is not a cuboid!");
  }
  return {m_points[0], m_points[1], m_points[2], m_points[3]};
}

__device__
CudaShapeInfo::SphereGeometry CudaShapeInfo::sphereGeometry() const {
  if (m_shape != ShapeInfo::GeometryShape::SPHERE) {
    raise("Cannot generate sphere geometry as shape is not a sphere!");
  }
  return {m_points[0], m_radius};
}

__device__
CudaShapeInfo::CylinderGeometry CudaShapeInfo::cylinderGeometry() const {
  if (m_shape != ShapeInfo::GeometryShape::CYLINDER) {
    raise("Cannot generate cylinder geometry as shape is not a cylinder!");
  }
  return {m_points[0], m_points[1], m_points[2], m_points[3], m_radius,
              m_height};
}

__device__
CudaShapeInfo::HollowCylinderGeometry CudaShapeInfo::hollowCylinderGeometry()
        const {
  if (m_shape != ShapeInfo::GeometryShape::HOLLOWCYLINDER) {
    raise("Cannot generate hollow cylinder geometry as shape is not a hollow "
          "cylinder!");
  }
  return {m_points[0], m_points[1], m_points[2], m_points[3], m_innerRadius,
              m_radius, m_height};
}

}
}
