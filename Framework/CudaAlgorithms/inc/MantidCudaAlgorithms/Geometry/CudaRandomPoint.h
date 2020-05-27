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
class CudaShapeInfo;
class CudaV3D;
class CudaBoundingBox;
class CudaMersenneTwister;
class CudaRule;

namespace CudaRandomPoint {

/**
 * Generate a point in a "local" Cylinder, ie one centred at (0,0) and aligned
 * to the z axis
 * @param prng the random number generator
 * @param radius the radius of the cylinder
 * @param height the height of the cylinder
 * @return a point in the local cylinder
 */
__device__
CudaV3D inLocalCylinder(const CudaMersenneTwister &prng,
                        const double radius,
                        const double height);

/**
 * Transforms a point first by a roation matrix of x, y and z and then a
 * translation.
 * @param pt point to transform
 * @param xRotation x rotation vector to apply
 * @param yRotation y rotation vector to apply
 * @param zRotation z rotation vector to apply
 * @param translation to apply
 * @return resulting point
 */
__device__
CudaV3D transformPoint(const CudaV3D &pt,
                       const CudaV3D &xRotation,
                       const CudaV3D &yRotation,
                       const CudaV3D &zRotation,
                       const CudaV3D &translation);

/**
 * Return a random point in a cuboid shape.
 * @param shapeInfo cuboid's shape info
 * @param rng a random number generate
 * @return a random point inside the cuboid
 */
__device__
CudaV3D inCuboid(const CudaShapeInfo &shapeInfo,
                 const CudaMersenneTwister &rng);

/**
 * Return a random point in cylinder.
 * @param shapeInfo cylinder's shape info
 * @param rng a random number generator
 * @return a point
 */
__device__
CudaV3D inCylinder(const CudaShapeInfo &shapeInfo,
                   const CudaMersenneTwister &rng);

/**
 * Return a random point in a hollow cylinder
 * @param shapeInfo hollow cylinder's shape info
 * @param rng a random number generator
 * @return a point
 */
__device__
CudaV3D inHollowCylinder(const CudaShapeInfo &shapeInfo,
                         const CudaMersenneTwister &rng);

/**
 * Return a random point in sphere.
 * @param shapeInfo sphere's shape info
 * @param rng a random number generator
 * @return a point
 */
__device__
CudaV3D inSphere(const CudaShapeInfo &shapeInfo,
                 const CudaMersenneTwister &rng);

/**
 * Return a random point in a known shape restricted by a bounding box.
 *
 * This could be called with one of the `inCylinder`, `inSphere`, ...
 * functions as the template argument.
 * @param shapeInfo a shape info
 * @param rng a random number generator
 * @param box a restricting box
 * @param maxAttempts number of attempts
 * @return a point or none if maxAttempts was exceeded
 */
template <CudaV3D (*T)(const CudaShapeInfo &shapeInfo,
                       const CudaMersenneTwister &rng)>
__device__
bool bounded(CudaV3D &pt,
             const CudaShapeInfo &shapeInfo,
             const CudaMersenneTwister &rng,
             const CudaBoundingBox &box);

/**
 * Return a random point in a generic shape limited by a bounding box.
 * @param object an object in which the point is generated
 * @param rng a random number generator
 * @param box a box restricting the point's volume
 * @param maxAttempts number of attempts to find a suitable point
 * @return a point or none if maxAttempts was exceeded
 */
__device__
bool bounded(CudaV3D &pt,
             const CudaRule &topRule,
             const CudaMersenneTwister &rng,
             const CudaBoundingBox &box);
}
}
}
