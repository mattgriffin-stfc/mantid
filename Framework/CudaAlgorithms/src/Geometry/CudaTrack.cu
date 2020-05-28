// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaTrack.h"

#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"
#include "MantidCudaAlgorithms/CudaVector.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaTrack::CudaTrack(const CudaV3D &startPt,
                     const CudaV3D &unitVector,
                     CudaVector<CudaLink> &links,
                     CudaVector<CudaIntersectionPoint> &surfPoints)
    : m_line(startPt, unitVector), m_links(links), m_surfPoints(surfPoints) {
  if (!unitVector.unitVector()) {
    raise("Failed to construct track: direction is not a unit vector.");
  }
}

__device__
void CudaTrack::addPoint(const Geometry::TrackDirection direction,
                         const CudaV3D &endPoint, const CudaMaterial * mat) {
  CudaIntersectionPoint newPoint(direction, endPoint, mat,
                                 endPoint.distance(m_line.getOrigin()));

  unsigned int i;
  for(i = 0; i < m_surfPoints.size(); i++) {
    if(newPoint < m_surfPoints[i]) {
      break;
    }
  }
  __syncthreads();

  m_surfPoints.insert(i, newPoint);
}

__device__
void CudaTrack::addLink(const CudaV3D &firstPoint, const CudaV3D &secondPoint,
                        const double distanceAlongTrack,
                        const CudaMaterial * mat) {
  // Process First Point
  CudaLink newLink(firstPoint, secondPoint, mat, distanceAlongTrack);

  unsigned int i;
  for(i = 0; i < m_links.size(); i++) {
    if(newLink < m_links[i]) {
      break;
    }
  }
  __syncthreads();

  m_links.insert(i, newLink);
}

__device__
void CudaTrack::buildLink() {
  // The surface points were added in order when they were built so no sorting
  // is required here.
  unsigned int ac = 0;
  unsigned int bc = ac;
  ++bc;
  // First point is not necessarily in an object
  // Process first point:
  for (ac = 0; ac < m_surfPoints.size(); ++ac) {
    const CudaIntersectionPoint &point = m_surfPoints[ac];
    if (point.direction == Geometry::TrackDirection::ENTERING) {
      break;
    }

    if (point.direction == Geometry::TrackDirection::LEAVING) {
      addLink(m_line.getOrigin(), point.endPoint, point.distFromStart,
              point.material);
    }
    bc += (bc < m_surfPoints.size());
  }

  __syncthreads();

  // have we now passed over all of the potential intersections without actually
  // hitting the object
  if (ac < m_surfPoints.size()) {
    CudaV3D workPt = m_surfPoints[ac].endPoint;       // last good point
    while (bc < m_surfPoints.size()) // Since bc > ac
    {
      const CudaIntersectionPoint &point1 = m_surfPoints[ac];
      const CudaIntersectionPoint &point2 = m_surfPoints[bc];

      if (point1.direction == Geometry::TrackDirection::ENTERING &&
          point2.direction == Geometry::TrackDirection::LEAVING) {
        // Touching surface / identical surface
        if (!CudaMathExtensions::isZero(point1.distFromStart -
                                        point2.distFromStart)) {
          // track leave ac into bc.
          addLink(point1.endPoint, point2.endPoint, point2.distFromStart,
                  point1.material);
        }
        // Points with intermediate void
        else {
          addLink(workPt, point1.endPoint, point1.distFromStart,
                  point1.material);
        }
        workPt = point2.endPoint;

        // incrementing ac twice: since processing pairs
        ++ac;
        ++ac;
        ++bc;
        bc += (bc < m_surfPoints.size());
      } else {
        ++ac;
        ++bc;
      }

      __syncthreads();

    }
  }

  __syncthreads();
  m_surfPoints.clear();
}

}
}
