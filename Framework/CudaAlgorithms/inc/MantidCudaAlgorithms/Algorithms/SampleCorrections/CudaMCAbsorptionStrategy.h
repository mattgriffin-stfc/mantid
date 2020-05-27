// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

#include "MantidKernel/DeltaEMode.h"

namespace Mantid {

using namespace Kernel;

namespace CudaAlgorithms {

class ICudaBeamProfile;
class CudaBoundingBox;
template <DeltaEMode::Type>
class CudaDetectorBinProvider;
class CudaCSGObject;
class CudaAlgorithmContext;
class CudaMersenneTwister;
template<DeltaEMode::Type>
struct CudaMCAbsorptionParameters;
class CudaMCGeometryBuffer;


template<DeltaEMode::Type etype>
class CudaMCAbsorptionStrategy {
public:
  /**
   * Construct the volume encompassing the sample + any environment kit. The
   * beam profile defines a bounding region for the sampling of the scattering
   * position.
   * @param sample A reference to a sample object that defines a valid shape
   * & material
   * @param activeRegion Restrict scattering point sampling to this region
   * @param maxScatterAttempts The maximum number of tries to generate a random
   * point within the object. [Default=5000]
   */
  CudaMCAbsorptionStrategy(const CudaMCAbsorptionParameters<etype> &d_params);

  void calculate(const cudaStream_t &algorithmStream = 0) const;

private:
  /// host
  const unsigned int m_blockSize;
  const dim3 m_gridDim;

  /// device
  double * md_output;
  CudaMersenneTwister * md_cudaRng;
  const ICudaBeamProfile * const * md_beamProfile;
  const CudaBoundingBox * md_scatterBounds;
  const CudaCSGObject * md_sample;
  const CudaCSGObject * md_environment;
  const CudaBoundingBox * md_activeRegion;
  const CudaDetectorBinProvider<etype> * md_eProvider;
  CudaAlgorithmContext * md_context;
  CudaMCGeometryBuffer * md_geometryBuffer;

};

}
}
