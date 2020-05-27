// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

namespace Mantid {

namespace Geometry {
enum class TrackDirection : int;
}

namespace CudaAlgorithms {

class CudaV3D;
class CudaShapeInfo;
class CudaBoundingBox;
class CudaMersenneTwister;
class CudaRule;
class CudaMaterial;
class CudaSurface;
class CudaTrack;
template <class T>
class CudaVector;

/**
 * The CudaCSGObject class, CUDA equivalent of CSGObject.
 */
class CudaCSGObject {
public:
  /**
   * Constructor for CudaCSGObject
   * @param shapeInfo descriptor for this shape to optimize geometric operations
   * @param sampleMaterial material the object is made of
   * @param topRule highest rule in the rule tree
   * @param surfaces that make up the object
   * @param nsurfaces number of surfaces in the object
   */
  __host__
  CudaCSGObject(const CudaShapeInfo * shapeInfo,
                const CudaMaterial * sampleMaterial,
                const CudaRule * const * topRule,
                const CudaSurface * const * surfaces,
                const unsigned int nsurfaces);

  /**
   * Generate a point within the bounds of this object
   * @param pt that was generated
   * @param prng the random number generator
   * @param activeRegion region a point can be within.
   * @return whether a point was successfully generated
   */
  __device__
  bool generatePointInObject(CudaV3D &pt,
                             const CudaMersenneTwister &prng,
                             const CudaBoundingBox &activeRegion) const;

  /**
   * Intercept the surfaces of this object.
   * @param cudaTrack the track to intercept this object with
   * @param distances between intercepted points
   * @param points intercepted
   * @return the number of surfaces intercepted
   */
  __device__
  int interceptSurface(CudaTrack &cudaTrack,
                       CudaVector<double> &distances,
                       CudaVector<CudaV3D> &points) const;

private:
  /**
   * Given a point and a direction,
   * @param point
   * @param uVec
   * @return ENTERING, LEAVING or INVALID
   */
  __device__
  Geometry::TrackDirection calcValidType(const CudaV3D &point,
                                         const CudaV3D &uVec) const;

  /// shape info descriptor for the object
  const CudaShapeInfo * md_shapeInfo;
  /// the material the object is constructed of
  const CudaMaterial * md_sampleMaterial;
  /// the rules that make up this object
  const CudaRule * const * md_topRule;
  /// the surfaces of this object
  const CudaSurface * const * md_surfaces;
};

}
}
