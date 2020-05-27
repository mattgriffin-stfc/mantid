// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/ICudaBeamProfile.h"

namespace Mantid {
namespace CudaAlgorithms {

/**
  Defines a flat, rectangular beam profile that has a width, height and center
  point. The profile is assumed infinitely thin.
*/
class CudaRectangularBeamProfile final : public ICudaBeamProfile {
public:
  /**
   * Construct a beam profile.
   * @param frame Defines the direction of the beam, up and horizontal
   * @param center V3D defining the central point of the rectangle
   * @param width Width of beam
   * @param height Height of beam
   */
  __device__
  CudaRectangularBeamProfile(unsigned short upIdx,
                             unsigned short beamIdx,
                             unsigned short horIdx,
                             const CudaV3D &center,
                             double width,
                             double height);

  __device__
  CudaRay generatePoint(const CudaMersenneTwister &prng) const override;
  __device__
  CudaRay generatePoint(const CudaMersenneTwister &prng,
                        const CudaBoundingBox &box) const override;
  __device__
  CudaBoundingBox defineActiveRegion(const CudaBoundingBox &) const override;

private:
  const unsigned short m_upIdx;
  const unsigned short m_beamIdx;
  const unsigned short m_horIdx;
  /// width of the beam
  const double m_width;
  /// height of the beam
  const double m_height;

  /// relative beam centre
  CudaV3D m_min;
  /// direction of the beam
  CudaV3D m_beamDir;
};

}
}
