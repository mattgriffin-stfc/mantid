// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2016 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidAlgorithms/DllConfig.h"
#include "MantidGeometry/Objects/BoundingBox.h"
#include "MantidKernel/Logger.h"
#include <boost/optional.hpp>

namespace Mantid {
namespace API {
class Sample;
}
namespace Geometry {
class IObject;
class SampleEnvironment;
class Track;
} // namespace Geometry

namespace Kernel {
class PseudoRandomNumberGenerator;
class V3D;
} // namespace Kernel
namespace Algorithms {
class IBeamProfile;

/**
  Defines a volume where interactions of Tracks and Objects can take place.
  Given an initial Track, end point & wavelengths it calculates the absorption
  correction factor.
*/
class MANTID_ALGORITHMS_DLL MCInteractionVolume {
public:
  MCInteractionVolume(const API::Sample &sample,
                      const Geometry::BoundingBox &activeRegion,
                      Kernel::Logger &logger,
                      const size_t maxScatterAttempts = 5000);
  // No creation from temporaries as we store a reference to the object in
  // the sample
  MCInteractionVolume(const API::Sample &&sample,
                      const Geometry::BoundingBox &&activeRegion,
                      Kernel::Logger &logger) = delete;

  const Geometry::BoundingBox &getBoundingBox() const;
  bool calculateBeforeAfterTrack(Kernel::PseudoRandomNumberGenerator &rng,
                                 const Kernel::V3D &startPos,
                                 const Kernel::V3D &endPos,
                                 Geometry::Track &beforeScatter,
                                 Geometry::Track &afterScatter);
  double calculateAbsorption(const Geometry::Track &beforeScatter,
                             const Geometry::Track &afterScatter,
                             double lambdaBefore, double lambdaAfter) const;
  void generateScatterPointStats();
  Kernel::V3D generatePoint(Kernel::PseudoRandomNumberGenerator &rng);

private:
  int getComponentIndex(Kernel::PseudoRandomNumberGenerator &rng);
  boost::optional<Kernel::V3D>
  generatePointInObjectByIndex(int componentIndex,
                               Kernel::PseudoRandomNumberGenerator &rng);
  void UpdateScatterPointCounts(int componentIndex);
  int m_sampleScatterPoints = 0;
  std::vector<int> m_envScatterPoints;
  const std::shared_ptr<Geometry::IObject> m_sample;
  const Geometry::SampleEnvironment *m_env;
  const Geometry::BoundingBox m_activeRegion;
  const size_t m_maxScatterAttempts;
  Kernel::Logger &m_logger;
};

} // namespace Algorithms
} // namespace Mantid
