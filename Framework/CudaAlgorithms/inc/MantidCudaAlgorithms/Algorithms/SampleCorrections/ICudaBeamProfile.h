// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"

namespace Mantid {
namespace CudaAlgorithms {

class CudaBoundingBox;
class CudaMersenneTwister;

/**
 * ICudaBeamProfile, equivalent of IBeamProfile.
 *
 * Base class for all beam profiles.
 */
class ICudaBeamProfile {
public:
  struct CudaRay {
    CudaV3D startPos;
    CudaV3D unitDir;
  };

  /**
   * Generate a random point within the beam profile using the supplied random
   * number source
   * @param rng A reference to a random number generator
   * @return An ICudaBeamProfile::Ray describing the start and direction
   */
  __device__
  virtual CudaRay generatePoint(const CudaMersenneTwister &rng) const = 0;

  /**
   * Generate a random point on the profile that is within the given bounding
   * area. If the point is outside the area then it is pulled to the boundary of
   * the bounding area.
   * @param rng A reference to a random number generator
   * @param bounds A reference to the bounding area that defines the maximum
   * allowed region for the generated point.
   * @return An ICudaBeamProfile::Ray describing the start and direction
   */
  __device__
  virtual CudaRay generatePoint(const CudaMersenneTwister &rng,
                                const CudaBoundingBox &box) const = 0;
  /**
   * Compute a region that defines how the beam illuminates the given sample/can
   * @param sample A reference to a sample object holding its shape
   * @return A BoundingBox defining the active region
   */
  __device__
  virtual CudaBoundingBox defineActiveRegion(
          const CudaBoundingBox &box) const = 0;
};

}
}
