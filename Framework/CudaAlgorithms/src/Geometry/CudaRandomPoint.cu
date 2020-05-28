// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaRandomPoint.h"

#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/Geometry/CudaShapeInfo.h"
#include "MantidCudaAlgorithms/Geometry/CudaBoundingBox.h"
#include "MantidCudaAlgorithms/Kernel/CudaMersenneTwister.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/Geometry/CudaRules.h"


namespace Mantid {
namespace CudaAlgorithms {
namespace CudaRandomPoint {

__device__
CudaV3D inCuboid(const CudaShapeInfo &shapeInfo,
                 const CudaMersenneTwister &prng) {
  const auto geometry = shapeInfo.cuboidGeometry();
  const double r1 = prng.nextValue();
  const double r2 = prng.nextValue();
  const double r3 = prng.nextValue();

  const CudaV3D &basis1 = geometry.leftFrontTop - geometry.leftFrontBottom;
  const CudaV3D &basis2 = geometry.leftBackBottom - geometry.leftFrontBottom;
  const CudaV3D &basis3 = geometry.rightFrontBottom - geometry.leftFrontBottom;

  return geometry.leftFrontBottom + (basis1 * r1 + basis2 * r2 + basis3 * r3);
}

__device__
CudaV3D inLocalCylinder(const CudaMersenneTwister &prng,
                        const double radius,
                        const double height) {
  const double r1 = prng.nextValue();
  const double r2 = prng.nextValue();
  const double r3 = prng.nextValue();
  const double polarFactor = 2. * r1;

  // The sqrt is needed for a uniform distribution of points.
  const double r = radius * sqrt(r2);
  const double z = height * r3;

  double y;
  double x;

  sincospi(polarFactor, &y, &x);

  return CudaV3D(x * r, y * r, z);
}

__device__
CudaV3D transformPoint(const CudaV3D &pt,
                       const CudaV3D &xRotation,
                       const CudaV3D &yRotation,
                       const CudaV3D &zRotation,
                       const CudaV3D &translation) {

  const double rx = pt.scalar_prod(xRotation);
  const double ry = pt.scalar_prod(yRotation);
  const double rz = pt.scalar_prod(zRotation);

  return CudaV3D(rx, ry, rz) + translation;
}

__device__
CudaV3D inCylinder(const CudaShapeInfo &shapeInfo,
                   const CudaMersenneTwister &prng) {
  const auto &geometry = shapeInfo.cylinderGeometry();

  const CudaV3D &localPt = inLocalCylinder(prng, geometry.radius,
                                           geometry.height);

  return transformPoint(localPt, geometry.xTransform, geometry.yTransform,
                        geometry.zTransform, geometry.centreOfBottomBase);
}

__device__
CudaV3D inHollowCylinder(const CudaShapeInfo &shapeInfo,
                         const CudaMersenneTwister &prng) {
  const auto &geometry = shapeInfo.hollowCylinderGeometry();

  const double r1 = prng.nextValue();
  const double r2 = prng.nextValue();
  const double r3 = prng.nextValue();
  const double polarFactor = 2. * r1;

  // The sqrt is needed for a uniform distribution of points.
  const double c1 = geometry.innerRadius * geometry.innerRadius;
  const double c2 = geometry.radius * geometry.radius;
  const double r = sqrt(c1 + (c2 - c1) * r2);
  const double z = geometry.height * r3;

  double y;
  double x;

  sincospi(polarFactor, &y, &x);

  CudaV3D localPt(x * r, y * r, z);

  return transformPoint(localPt, geometry.xTransform, geometry.yTransform,
                        geometry.zTransform, geometry.centreOfBottomBase);
}

__device__
CudaV3D inSphere(const CudaShapeInfo &shapeInfo,
                 const CudaMersenneTwister &prng) {
  const auto geometry = shapeInfo.sphereGeometry();
  const double r1 = prng.nextValue();
  const double r2 = prng.nextValue();
  const double r3 = prng.nextValue();
  const double azimuthalFactor = 2. * r1;

  double sinAzimuthal;
  double cosAzimuthal;

  sincospi(azimuthalFactor, &sinAzimuthal, &cosAzimuthal);

  double sinPolar;
  double cosPolar;

  // The acos is needed for a uniform distribution of points.
  const double polar = acos(2. * r2 - 1.);

  sincos(polar, &sinPolar, &cosPolar);

  const double r = r3 * geometry.radius;
  const double x = r * cosAzimuthal * sinPolar;
  const double y = r * sinAzimuthal * sinPolar;
  const double z = r * cosPolar;

  return geometry.centre + CudaV3D(x, y, z);
}

__device__
bool bounded(CudaV3D &pt,
             const CudaRule &topRule,
             const CudaMersenneTwister &prng,
             const CudaBoundingBox &box) {

  const double r1 = prng.nextValue();
  const double r2 = prng.nextValue();
  const double r3 = prng.nextValue();
  pt = box.generatePointInside(r1, r2, r3);
  return topRule.isValid(pt);
}


template <CudaV3D (*randomInShape)(const CudaShapeInfo &,
                                   const CudaMersenneTwister &)>
__device__
bool bounded(CudaV3D &pt,
             const CudaShapeInfo &shapeInfo,
             const CudaMersenneTwister &prng,
             const CudaBoundingBox &box) {

  pt = randomInShape(shapeInfo, prng);
  return box.isPointInside(pt);
}

// forward declarations
template __device__
bool bounded<inCylinder>(CudaV3D &, const CudaShapeInfo &,
                         const CudaMersenneTwister &,
                         const CudaBoundingBox &);

template __device__
bool bounded<inHollowCylinder>(CudaV3D &, const CudaShapeInfo &,
                               const CudaMersenneTwister &,
                               const CudaBoundingBox &);

template __device__
bool bounded<inCuboid>(CudaV3D &, const CudaShapeInfo &,
                       const CudaMersenneTwister &, const CudaBoundingBox &);

template __device__
bool bounded<inSphere>(CudaV3D &, const CudaShapeInfo &,
                       const CudaMersenneTwister &, const CudaBoundingBox &);
}
}
}
