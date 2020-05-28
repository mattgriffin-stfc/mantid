// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace Mantid {

namespace Geometry {
class ReferenceFrame;
class IComponent;
class BoundingBox;
}

namespace CudaAlgorithms {
class ICudaBeamProfile;
class CudaBoundingBox;

class CudaBeamProfileFactory {
public:

  void createBeamProfile(ICudaBeamProfile ** &beamProfile,
                         std::shared_ptr<const Geometry::ReferenceFrame> frame,
                         std::shared_ptr<const Geometry::IComponent> source,
                         const Geometry::BoundingBox &scatterBounds,
                         const cudaStream_t &cudaStream) const;

  /**
   * Defines the active region of a sample based upon the samples scatter bounds
   * and the profile of the beam.
   *
   * Method is asynchronous, requires synchronization after calling.
   *
   * @param activeRegion   empty device pointer that will hold the active region
   *                       (device ptr)
   * @param beamProfile    the profile of the neutron beam (device ptr)
   * @param scatterBounds  the bounds of the sample and its environment to
   *                       scatter in (device ptr)
   * @param cudaStream     stream to execute the operation, defaults to 0.
   */
  void defineActiveRegion(CudaBoundingBox * &activeRegion,
                          const ICudaBeamProfile * const * beamProfile,
                          const CudaBoundingBox * scatterBounds,
                          const cudaStream_t &cudaStream = 0) const;
};
}
}
