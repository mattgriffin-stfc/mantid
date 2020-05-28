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
class CudaLine;
class CudaSurface;
class CudaQuadratic;
class CudaPlane;
class CudaCylinder;
class CudaSphere;
class CudaCone;
class CudaV3D;

template<typename T>
class CudaVector;

/**
 * The CudaLineIntersectVisit class (cut down, CUDA equivalent for
 * Geometry::LineIntersectVisit)
 *
 * Constructable on device, usable on device.
 *
 * Creates interaction with a line and surface.
 */
class CudaLineIntersectVisit {
public:
  /**
   * Constructor for CudaLineIntersectVisit
   * @param line of intersection
   * @param DOut vector of distances between intersection points
   * @param PtOut vector of intersection points
   */
  __device__
  CudaLineIntersectVisit(const CudaLine &line, CudaVector<double> &DOut,
                         CudaVector<CudaV3D> &PtOut);

  /**
    Process an intersect track
    @param Surf :: Surface to use int line Interesect
  */
  __device__
  void Accept(const CudaQuadratic &surf);
  __device__
  void Accept(const CudaPlane &surf);
  __device__
  void Accept(const CudaCylinder &surf);
  __device__
  void Accept(const CudaSphere &surf);
  __device__
  void Accept(const CudaCone &surf);

  /**
   * @return the intersection points
   */
  __inline__ __device__
  const CudaVector<CudaV3D> &getPoints() const { return PtOut; }

  /**
   * @brief procTrack builds the links in a track
   * @return vector of distances between the track links
   */
  __device__
  const CudaVector<double> &procTrack();

private:
  /// vector of distances between the intersection points
  CudaVector<double> &DOut;
  /// vector of the intersection points along the line
  CudaVector<CudaV3D> &PtOut;
  /// the tracked line to intersect
  const CudaLine &ATrack;
};

}
}
