// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

#include "MantidGeometry/Objects/Track.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/Geometry/CudaLine.h"
#include "MantidCudaAlgorithms/CudaMathExtensions.h"

namespace Mantid {
namespace CudaAlgorithms {
class CudaMaterial;
template <typename T>
class CudaVector;

/**
\struct Link
\author S. Ansell
\author M. Gigg, Tessella plc
\brief For a leg of a track
*/
struct CudaLink {
  /**
   * Constuctor
   * @param entry :: Kernel::V3D point to start
   * @param exit :: Kernel::V3D point to end track
   * @param totalDistance :: Total distance from start of track
   * @param obj :: A reference to the object that was intersected
   * @param compID :: An optional component identifier for the physical object
   * hit. (Default=NULL)
   */
  __inline__ __device__
  CudaLink(const CudaV3D &entry, const CudaV3D &exit, const CudaMaterial * mat,
           const double totalDistance)
      : distInsideObject(entry.distance(exit)),
        distFromStart(totalDistance),
        material(mat) {}

  /// Less than operator
  __inline__ __device__
  bool operator<(const CudaLink &other) const {
    return distFromStart < other.distFromStart;
  }

  /** @name Attributes. */
  double distInsideObject; ///< Total distance covered inside object
  double distFromStart;    ///< Total distance from track beginning
  const CudaMaterial * material;   ///< The object that was intersected
};

/**
 * Stores a point of intersection along a track. The component intersected is
 * linked using its ComponentID.
 *
 * Ordering for IntersectionPoint is special since we need that when dist is
 * close that the +/- flag is taken into account.
 */
struct CudaIntersectionPoint {
  /**
   * Constuctor
   * @param direction :: Indicates the direction of travel of the track with
   * respect to the object: +1 is entering, -1 is leaving.
   * @param end :: The end point for this partial segment
   * @param distFromStartOfTrack :: Total distance from start of track
   * @param compID :: An optional unique ID marking the component intersected.
   * (Default=NULL)
   * @param obj :: A reference to the object that was intersected
*/
  __inline__ __device__
  CudaIntersectionPoint(const Geometry::TrackDirection direction,
                        const CudaV3D &end,
                        const CudaMaterial * mat,
                        const double distFromStartOfTrack)
      : material(mat), distFromStart(distFromStartOfTrack),
        endPoint(end), direction(direction) {}

  /**
   * A IntersectionPoint is less-than another if either
   * (a) the difference in distances is greater than the tolerance and this
   *distance is less than the other or
   * (b) the distance is less than the other and this point is defined as an
   *exit point
   *
   * @param other :: IntersectionPoint object to compare
   * @return True if the object is considered less than, otherwise false.
   */
  __inline__ __device__
  bool operator<(const CudaIntersectionPoint &other) const {
      return CudaMathExtensions::isZero(distFromStart - other.distFromStart) ?
                  ((int)direction) < ((int)other.direction)
                : distFromStart < other.distFromStart;
  }

  const CudaMaterial * material;
  double distFromStart;
  CudaV3D endPoint;
  Geometry::TrackDirection direction;
};

/**
 * Defines a track as a start point and a direction. Intersections are stored as
 * ordered lists of links from the start point to the exit point.
 */
class CudaTrack {
public:
  /**
   * Constructor for CudaTrack
   * @param startPt :: Initial point
   * @param unitVector :: Directional vector. It must be unit vector.
   */
  __device__
  CudaTrack(const CudaV3D &startPt,
            const CudaV3D &unitVector,
            CudaVector<CudaLink> &links,
            CudaVector<CudaIntersectionPoint> &surfPoints);

  /**
   * This adds a whole segment to the track : This currently assumes that links
   * are added in order
   * @param firstPoint :: first Point
   * @param secondPoint :: second Point
   * @param distanceAlongTrack :: Distance along track
   * @param obj :: A reference to the object that was intersected
   * @param compID :: ID of the component that this link is about (Default=NULL)
   */
  __device__
  void addPoint(const Geometry::TrackDirection direction,
                const CudaV3D &endPoint,
                const CudaMaterial * mat);

  /**
   * Builds a set of linking track components.
   * This version deals with touching surfaces
  */
  __device__
  void addLink(const CudaV3D &firstPoint, const CudaV3D &secondPoint,
               const double distanceAlongTrack, const CudaMaterial * mat);

  /// Construct links between added points
  __device__
  void buildLink();

  /// Returns the starting point
  __inline__ __device__
  const CudaV3D &startPoint() const { return m_line.getOrigin(); }

  /// Returns the direction
  __inline__ __device__
  const CudaV3D &direction() const { return m_line.getDirect(); }

  /// Return reference to the tracks line
  __inline__ __device__
  const CudaLine &getLine() const { return m_line; }

  /// Return reference to the links
  __inline__ __device__
  const CudaVector<CudaLink> &getlinks() const { return m_links; }

private:
  /// Line object containing origin and direction
  const CudaLine m_line;
  /// Track units
  CudaVector<CudaLink> &m_links;
  /// Intersection points
  CudaVector<CudaIntersectionPoint> &m_surfPoints;
};

}
}
