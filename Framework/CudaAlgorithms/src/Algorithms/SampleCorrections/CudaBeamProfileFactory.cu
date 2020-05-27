// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaBeamProfileFactory.h"

#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/ICudaBeamProfile.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaRectangularBeamProfile.h"
#include "MantidCudaAlgorithms/Geometry/CudaBoundingBox.h"
#include "MantidCudaAlgorithms/CudaGuard.h"
#include "MantidGeometry/Instrument/ReferenceFrame.h"
#include "MantidGeometry/IComponent.h"

namespace Mantid {

using namespace Geometry;

namespace CudaAlgorithms {

__global__
void createBeamProfileKernel(ICudaBeamProfile ** beamProfile,
                             const unsigned short upIdx,
                             const unsigned short beamIdx,
                             const unsigned short horIdx,
                             const double3 centre,
                             const double width,
                             const double height) {

  *beamProfile = new CudaRectangularBeamProfile(upIdx, beamIdx, horIdx,
                                                CudaV3D(centre.x, centre.y,
                                                        centre.z),
                                                width, height);
}

void CudaBeamProfileFactory::createBeamProfile(
        ICudaBeamProfile ** &beamProfile,
        std::shared_ptr<const ReferenceFrame> frame,
        std::shared_ptr<const IComponent> source,
        const BoundingBox &scatterBounds,
        const cudaStream_t &cudaStream) const {

  auto beamWidthParam = source->getNumberParameter("beam-width");
  auto beamHeightParam = source->getNumberParameter("beam-height");
  double beamWidth(-1.0), beamHeight(-1.0);
  if (beamWidthParam.size() == 1 && beamHeightParam.size() == 1) {
    beamWidth = beamWidthParam[0];
    beamHeight = beamHeightParam[0];
  } else {
    const auto bbox = scatterBounds.width();
    beamWidth = bbox[frame->pointingHorizontal()];
    beamHeight = bbox[frame->pointingUp()];
  }

  double3 centre = make_double3(source->getPos().X(), source->getPos().Y(),
                                source->getPos().Z());

  CUDA_GUARD(cudaMalloc(&beamProfile, sizeof(ICudaBeamProfile*)));

  createBeamProfileKernel<<<1, 1, 0, cudaStream>>>(beamProfile,
                                                   frame->pointingUp(),
                                                   frame->pointingAlongBeam(),
                                                   frame->pointingHorizontal(),
                                                   centre, beamWidth,
                                                   beamHeight);
}

__global__
void defineActiveRegionKernel(CudaBoundingBox * activeRegion,
                              const ICudaBeamProfile * const * beamProfile,
                              const CudaBoundingBox * scatterBounds) {

  *activeRegion = (*beamProfile)->defineActiveRegion(*scatterBounds);
}

void CudaBeamProfileFactory::defineActiveRegion(
        CudaBoundingBox * &activeRegion,
        const ICudaBeamProfile * const * beamProfile,
        const CudaBoundingBox * scatterBounds,
        const cudaStream_t &cudaStream) const {

  CUDA_GUARD(cudaMalloc(&activeRegion, sizeof(CudaBoundingBox)));

  defineActiveRegionKernel<<<1, 1, 0, cudaStream>>>(activeRegion, beamProfile,
                                                    scatterBounds);
}
}
}
