// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaRectangularBeamProfile.h"

#include "MantidCudaAlgorithms/Geometry/CudaBoundingBox.h"
#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/Kernel/CudaMersenneTwister.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaRectangularBeamProfile::CudaRectangularBeamProfile(
        unsigned short upIdx,
        unsigned short beamIdx,
        unsigned short horIdx,
        const CudaV3D &center,
        double width,
        double height)
    : ICudaBeamProfile(), m_upIdx(upIdx), m_beamIdx(beamIdx), m_horIdx(horIdx),
      m_width(width), m_height(height), m_min() {

  m_min[m_upIdx] = fma(-0.5, height, center[m_upIdx]);
  m_min[m_horIdx] = fma(-0.5, width, center[m_horIdx]);
  m_min[m_beamIdx] = center[m_beamIdx];
}

__device__
ICudaBeamProfile::CudaRay CudaRectangularBeamProfile::generatePoint(
        const CudaMersenneTwister &rng) const {
  CudaV3D pt;
  pt[m_upIdx] = fma(rng.nextValue(), m_height, m_min[m_upIdx]);
  pt[m_horIdx] = fma(rng.nextValue(), m_width, m_min[m_horIdx]);
  pt[m_beamIdx] = m_min[m_beamIdx];
  return {pt, m_beamDir};
}

__device__
ICudaBeamProfile::CudaRay CudaRectangularBeamProfile::generatePoint(
        const CudaMersenneTwister &rng, const CudaBoundingBox &bounds) const {
  auto rngRay = generatePoint(rng);
  auto &rngPt = rngRay.startPos;

  const CudaV3D &minBound = bounds.minPoint();
  const CudaV3D &maxBound = bounds.maxPoint();

  rngPt[m_upIdx] = CudaMathExtensions::fclamp(rngPt[m_upIdx], minBound[m_upIdx],
                                              maxBound[m_upIdx]);
  rngPt[m_horIdx] = CudaMathExtensions::fclamp(rngPt[m_horIdx], minBound[m_horIdx],
                                               maxBound[m_horIdx]);
  return rngRay;
}

__device__
CudaBoundingBox CudaRectangularBeamProfile::defineActiveRegion(
        const CudaBoundingBox &sampleBox) const {
  // In the beam direction use the maximum sample extent other wise restrict
  // the active region to the width/height of beam
  const auto &sampleMin(sampleBox.minPoint());
  const auto &sampleMax(sampleBox.maxPoint());
  CudaV3D minPoint, maxPoint;
  minPoint[m_horIdx] = fmax(sampleMin[m_horIdx], m_min[m_horIdx]);
  maxPoint[m_horIdx] = fmin(sampleMax[m_horIdx], m_min[m_horIdx] + m_width);
  minPoint[m_upIdx] = fmax(sampleMin[m_upIdx], m_min[m_upIdx]);
  maxPoint[m_upIdx] = fmin(sampleMax[m_upIdx], m_min[m_upIdx] + m_height);
  minPoint[m_beamIdx] = sampleMin[m_beamIdx];
  maxPoint[m_beamIdx] = sampleMax[m_beamIdx];

  return CudaBoundingBox(maxPoint.X(), maxPoint.Y(), maxPoint.Z(),
                         minPoint.X(), minPoint.Y(), minPoint.Z());
}

}
}
