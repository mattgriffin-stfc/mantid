// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionSetup.h"

#include <iostream>

#include "MantidAPI/Sample.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidGeometry/Instrument/SampleEnvironment.h"
#include "MantidGeometry/Instrument.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCGeometryBuffer.h"
#include "MantidCudaAlgorithms/Geometry/CudaBoundingBox.h"
#include "MantidCudaAlgorithms/Geometry/CudaCSGObject.h"
#include "MantidCudaAlgorithms/Kernel/CudaMersenneTwister.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"
#include "MantidCudaAlgorithms/CudaGuard.h"

namespace Mantid {
using namespace Kernel;
using namespace Geometry;
using namespace API;

namespace CudaAlgorithms {

template<DeltaEMode::Type etype>
CudaCSGObject * CudaMCAbsorptionSetup<etype>::createEnvironment(
        const Geometry::SampleEnvironment &env,
        const cudaStream_t &algorithmStream) {

  const size_t numberOfElements = env.nelements();
  const size_t environmentSize = sizeof(CudaCSGObject) * numberOfElements;
  CudaCSGObject * d_environments;

  CUDA_GUARD(cudaMalloc(&d_environments, environmentSize));

  for (size_t i = 0; i < numberOfElements; i++) {
    objectFactory.createCSGObject(d_environments + i, env.getComponent(i),
                                  algorithmStream);
  }

  return d_environments;
}

template<DeltaEMode::Type etype>
CudaCSGObject * CudaMCAbsorptionSetup<etype>::createSample(
        const API::Sample &sample,
        const cudaStream_t &algorithmStream) {

  const size_t sampleSize = sizeof(CudaCSGObject);
  CudaCSGObject * d_sample;
  CUDA_GUARD(cudaMalloc(&d_sample, sampleSize));

  objectFactory.createCSGObject(d_sample, sample.getShape(), algorithmStream);

  return d_sample;
}

template<DeltaEMode::Type etype>
CudaMCAbsorptionSetup<etype>::CudaMCAbsorptionSetup(
        const API::MatrixWorkspace &instrumentWS,
        const CudaAlgorithmContext &context,
        const unsigned int nhists,
        const unsigned int nbins,
        const unsigned int nevents,
        const unsigned int blockSize,
        const unsigned int maxScatterPtAttempts,
        const unsigned int geometryBuffer,
        const int seed,
        const cudaDeviceProp &deviceProperties,
        const cudaStream_t &cudaStream)
    : eProviderFactory(CudaDetectorBinProviderSetup<etype>(instrumentWS,
                                                           nbins,
                                                           nhists,
                                                           deviceProperties,
                                                           cudaStream)) {

  const unsigned int multiprocessors = deviceProperties.multiProcessorCount;
  const unsigned int cores = _ConvertSMVer2Cores(deviceProperties.major,
                                                 deviceProperties.minor);
  const unsigned int warpSize = deviceProperties.warpSize;

  auto sample = instrumentWS.sample();

  const unsigned int maxHists = deviceProperties.maxGridSize[2];

  if (nhists > maxHists) {
    std::ostringstream error;
    error << "Exceeded max number of hists: "
          << nhists << "/" << maxHists
          << std::endl;
    throw std::runtime_error(error.str());
  }

  const unsigned int maxBins = deviceProperties.maxGridSize[1];

  if (nbins > maxBins) {
    std::ostringstream error;
    error << "Exceeded max number of bins: "
          << nbins << "/" << maxBins
          << std::endl;
    throw std::runtime_error(error.str());
  }

  const unsigned int maxEvents = deviceProperties.maxGridSize[0] * blockSize;

  if (nevents > maxEvents) {
    std::ostringstream error;
    error << "Exceeded max number of nevents: "
          << nevents << "/" << maxEvents
          << std::endl;
    throw std::runtime_error(error.str());
  }

  const size_t cudaBoundingBoxSize = sizeof(CudaBoundingBox);

  BoundingBox scatterBounds = sample.getShape().getBoundingBox();
  CudaBoundingBox h_scatterBounds(scatterBounds.xMax(), scatterBounds.yMax(),
                                  scatterBounds.zMax(), scatterBounds.xMin(),
                                  scatterBounds.yMin(), scatterBounds.zMin());

  CUDA_GUARD(cudaMalloc(&m_deviceParams.scatterBounds, cudaBoundingBoxSize));
  CUDA_GUARD(cudaMemcpyAsync(m_deviceParams.scatterBounds, &h_scatterBounds,
                             cudaBoundingBoxSize, cudaMemcpyHostToDevice,
                             cudaStream));

  Geometry::BoundingBox scatterEnvBounds = sample.getShape().getBoundingBox();
  if (sample.hasEnvironment()) {
    scatterEnvBounds.grow(sample.getEnvironment().boundingBox());
  }
  h_scatterBounds = CudaBoundingBox(scatterEnvBounds.xMax(),
                                    scatterEnvBounds.yMax(),
                                    scatterEnvBounds.zMax(),
                                    scatterEnvBounds.xMin(),
                                    scatterEnvBounds.yMin(),
                                    scatterEnvBounds.zMin());

  CUDA_GUARD(cudaMalloc(&md_environmentBounds, cudaBoundingBoxSize));
  CUDA_GUARD(cudaMemcpyAsync(md_environmentBounds, &h_scatterBounds,
                             cudaBoundingBoxSize, cudaMemcpyHostToDevice,
                             cudaStream));

  const auto instrument = instrumentWS.getInstrument();

  beamFactory.createBeamProfile(m_deviceParams.beamProfile,
                                instrument->getReferenceFrame(),
                                instrument->getSource(), scatterBounds,
                                cudaStream);

  beamFactory.defineActiveRegion(m_deviceParams.activeRegion,
                                 m_deviceParams.beamProfile,
                                 md_environmentBounds, cudaStream);

  m_deviceParams.sample = createSample(sample, cudaStream);
  std::cerr << m_deviceParams.sample << std::endl;

  if (sample.hasEnvironment()) {
    auto &environment = sample.getEnvironment();
    m_deviceParams.environment = createEnvironment(environment, cudaStream);
    m_deviceParams.nenv = static_cast<unsigned int>(environment.nelements());
  } else {
    m_deviceParams.nenv = 0;
  }

  // create geometry buffer
  h_geomBuffer = new CudaMCGeometryBuffer(geometryBuffer, warpSize, cores,
                                          multiprocessors);

  const size_t geomBufferSize = sizeof(CudaMCGeometryBuffer);
  CUDA_GUARD(cudaMalloc(&m_deviceParams.geometryBuffer, geomBufferSize));
  CUDA_GUARD(cudaMemcpy(m_deviceParams.geometryBuffer,
                        h_geomBuffer,
                        geomBufferSize,
                        cudaMemcpyHostToDevice));


  const size_t mersenneSize = sizeof(CudaMersenneTwister);
  mt = new CudaMersenneTwister(seed, cores, multiprocessors);
  CUDA_GUARD(cudaMalloc(&m_deviceParams.prng, mersenneSize));
  CUDA_GUARD(cudaMemcpy(m_deviceParams.prng, mt, mersenneSize,
                        cudaMemcpyHostToDevice));

  // allocate outputs
  const size_t outputSize = sizeof(double) * nbins * nhists;
  CUDA_GUARD(cudaMalloc(&m_deviceParams.output, outputSize));
  CUDA_GUARD(cudaMemset(m_deviceParams.output, 0, outputSize));

  // allocate context
  const size_t algorithmContextSize = sizeof(CudaAlgorithmContext);
  CUDA_GUARD(cudaMalloc(&m_deviceParams.context, algorithmContextSize));
  CUDA_GUARD(cudaMemcpy(m_deviceParams.context, &context, algorithmContextSize,
                        cudaMemcpyHostToDevice));

  m_deviceParams.eProvider = eProviderFactory.getProvider();
  m_deviceParams.nhists = nhists;
  m_deviceParams.nbins = nbins;
  m_deviceParams.nevents = nevents;
  m_deviceParams.blockSize = blockSize;
  m_deviceParams.maxScatterPtAttempts = maxScatterPtAttempts;
}

/**
 * @brief CudaMCAbsorptionSetup::~CudaMCAbsorptionSetup
 * Use CUDA_SOFT_GUARD so exceptions aren't thrown from deconstructor.
 */
template<DeltaEMode::Type etype>
CudaMCAbsorptionSetup<etype>::~CudaMCAbsorptionSetup() {
  delete mt;
  delete h_geomBuffer;

  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.output));

  // free sample
  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.sample));
  CUDA_SOFT_GUARD(cudaFree(md_environmentBounds));

  // free environment  
  if (m_deviceParams.nenv > 0) {
    CUDA_SOFT_GUARD(cudaFree(m_deviceParams.environment));
  }

  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.scatterBounds));
  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.activeRegion));
  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.beamProfile));
  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.prng));

  // free the buffers
  CUDA_SOFT_GUARD(cudaFree(m_deviceParams.geometryBuffer));
}

// forward declaration
template class CudaMCAbsorptionSetup<DeltaEMode::Direct>;
template class CudaMCAbsorptionSetup<DeltaEMode::Indirect>;
template class CudaMCAbsorptionSetup<DeltaEMode::Elastic>;

}
}
