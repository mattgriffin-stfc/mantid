// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaCSGObject.h"

#include "MantidGeometry/Objects/Track.h"

#include "MantidCudaAlgorithms/Geometry/CudaBoundingBox.h"
#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"
#include "MantidCudaAlgorithms/Geometry/CudaRandomPoint.h"
#include "MantidCudaAlgorithms/Geometry/CudaRules.h"
#include "MantidCudaAlgorithms/Geometry/CudaShapeInfo.h"
#include "MantidCudaAlgorithms/Geometry/CudaSurface.h"
#include "MantidCudaAlgorithms/Geometry/CudaTrack.h"
#include "MantidCudaAlgorithms/Kernel/CudaMaterial.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/Kernel/CudaMersenneTwister.h"
#include "MantidCudaAlgorithms/CudaVector.h"
#include "MantidCudaAlgorithms/CudaGuard.h"

namespace Mantid {

using namespace Geometry::detail;

namespace CudaAlgorithms {

__constant__ double VALID_INTERCEPT_POINT_SHIFT = 2.5e-05;
__constant__ unsigned int N_SURFACES;

__host__
CudaCSGObject::CudaCSGObject(const CudaShapeInfo * d_shapeInfo,
                             const CudaMaterial * d_sampleMaterial,
                             const CudaRule * const * d_topRule,
                             const CudaSurface * const * d_surfaces,
                             const unsigned int nsurfaces)
    : md_shapeInfo(d_shapeInfo),
      md_sampleMaterial(d_sampleMaterial),
      md_topRule(d_topRule),
      md_surfaces(d_surfaces) {
  CUDA_GUARD(cudaMemcpyToSymbol(N_SURFACES, &nsurfaces, sizeof(int)));
}

__device__
bool CudaCSGObject::generatePointInObject(
        CudaV3D &pt,
        const CudaMersenneTwister &prng,
        const CudaBoundingBox &activeRegion) const {

  const CudaShapeInfo &shapeInfo = *md_shapeInfo;

  switch (shapeInfo.shape()) {
    case ShapeInfo::GeometryShape::CUBOID:
      return CudaRandomPoint::bounded<CudaRandomPoint::inCuboid>(pt, shapeInfo,
                  prng, activeRegion);

    case ShapeInfo::GeometryShape::CYLINDER:
      return CudaRandomPoint::bounded<CudaRandomPoint::inCylinder>(pt,
                  shapeInfo, prng, activeRegion);

    case ShapeInfo::GeometryShape::HOLLOWCYLINDER:
      return CudaRandomPoint::bounded<CudaRandomPoint::inHollowCylinder>(pt,
                  shapeInfo, prng, activeRegion);

    case ShapeInfo::GeometryShape::SPHERE:
      return CudaRandomPoint::bounded<CudaRandomPoint::inSphere>(pt, shapeInfo,
                  prng, activeRegion);

    default:
      return CudaRandomPoint::bounded(pt, *md_topRule[0], prng, activeRegion);
  }
}

__device__
int CudaCSGObject::interceptSurface(CudaTrack &track,
                                    CudaVector<double> &distances,
                                    CudaVector<CudaV3D> &points) const {
  // Number of intersections original track
  int originalCount = track.getlinks().size();

  // Loop over all the surfaces.
  CudaLineIntersectVisit LI(track.getLine(), distances, points);

  for (unsigned int i = 0; i < N_SURFACES; i++) {
    const CudaSurface * surface = md_surfaces[i];

    surface->acceptVisitor(LI);
    __syncthreads();
  }

  const CudaVector<double> &dPoints = LI.procTrack();
  const CudaVector<CudaV3D> &IPoints = LI.getPoints();

  for (unsigned int i = 0; i < IPoints.size(); ++i) {
    if (dPoints[i] > 0.0) // only interested in forward going points
    {
      // Is the point and enterance/exit Point
      const CudaV3D &pt = IPoints[i];
      const Geometry::TrackDirection flag = calcValidType(pt,
                                                          track.direction());

      if (flag != Geometry::TrackDirection::INVALID) {
        track.addPoint(flag, pt, md_sampleMaterial);
      }
    }
  }

  __syncthreads();
  track.buildLink();

  __syncthreads();

  // Return number of track segments added
  return (track.getlinks().size() - originalCount);
}

__device__
Geometry::TrackDirection CudaCSGObject::calcValidType(
        const CudaV3D &point,
        const CudaV3D &uVec) const {

  const CudaRule &topRule = *md_topRule[0];
  const CudaV3D &shift(uVec * VALID_INTERCEPT_POINT_SHIFT);

  const int flagA = topRule.isValid(point - shift);
  __syncthreads();
  const int flagB = topRule.isValid(point + shift);
  __syncthreads();

  Geometry::TrackDirection direc;
  if (!(flagA ^ flagB))
    direc = Geometry::TrackDirection::INVALID;
  else if (flagA)
    direc = Geometry::TrackDirection::LEAVING;
  else
    direc = Geometry::TrackDirection::ENTERING;

  __syncthreads();

  return direc;
}

}
}
