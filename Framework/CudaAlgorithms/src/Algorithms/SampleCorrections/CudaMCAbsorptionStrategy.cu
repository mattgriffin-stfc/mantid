// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionStrategy.h"

#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBin.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProvider.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionParameters.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCGeometryBuffer.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/ICudaBeamProfile.h"
#include "MantidCudaAlgorithms/Geometry/CudaCSGObject.h"
#include "MantidCudaAlgorithms/Geometry/CudaTrack.h"
#include "MantidCudaAlgorithms/Kernel/CudaMaterial.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"
#include "MantidCudaAlgorithms/CudaGuard.h"
#include "MantidCudaAlgorithms/CudaReduce.h"
#include "MantidCudaAlgorithms/CudaVector.h"

namespace Mantid {
namespace CudaAlgorithms {

__constant__ unsigned int N_EVENTS;
__constant__ unsigned int N_BINS;
__constant__ unsigned int N_HISTS;
__constant__ unsigned int N_ENV;
__constant__ unsigned int N_SCATTER_OBJS;
__constant__ unsigned int MAX_SCATTER_ATTEMPTS;

/**
 * Calculate total attenuation for a track
 */
__inline__ __device__
double calculateAttenuation(const CudaTrack &path, const double lambda) {
  double factor(1.0);

  const CudaVector<CudaLink> &links = path.getlinks();

  for (unsigned int i = 0; i < links.size(); i++) {
    const CudaLink &segment = links[i];
    const double length = segment.distInsideObject;

    factor *= segment.material->attenuation(length, lambda);
  }

  __syncthreads();
  return factor;
}

template<DeltaEMode::Type etype>
CudaMCAbsorptionStrategy<etype>::CudaMCAbsorptionStrategy(
        const CudaMCAbsorptionParameters<etype> &d_params)
    : m_blockSize(d_params.blockSize),
      m_gridDim(make_uint3((d_params.nevents + m_blockSize - 1) / m_blockSize,
                           d_params.nbins,
                           d_params.nhists)),
      md_output(d_params.output),
      md_cudaRng(d_params.prng),
      md_beamProfile(d_params.beamProfile),
      md_scatterBounds(d_params.scatterBounds),
      md_sample(d_params.sample),
      md_environment(d_params.environment),
      md_activeRegion(d_params.activeRegion),
      md_eProvider(d_params.eProvider),
      md_context(d_params.context),
      md_geometryBuffer(d_params.geometryBuffer) {

  CUDA_GUARD(cudaMemcpyToSymbol(N_EVENTS, &d_params.nevents, sizeof(int)));
  CUDA_GUARD(cudaMemcpyToSymbol(N_ENV, &d_params.nenv, sizeof(int)));
  CUDA_GUARD(cudaMemcpyToSymbol(N_BINS, &d_params.nbins, sizeof(int)));
  CUDA_GUARD(cudaMemcpyToSymbol(N_HISTS, &d_params.nhists, sizeof(int)));
  CUDA_GUARD(cudaMemcpyToSymbol(MAX_SCATTER_ATTEMPTS,
                                &d_params.maxScatterPtAttempts, sizeof(int)));
}

__device__
CudaV3D generatePoint(const CudaCSGObject &sample,
                      const CudaCSGObject * environment,
                      const CudaMersenneTwister &prng,
                      const CudaBoundingBox &activeRegion,
                      const unsigned int eventId) {

  unsigned int warpSequence = (eventId / warpSize);
  unsigned int scatterObj = warpSequence % N_SCATTER_OBJS;
  CudaV3D pt;

  unsigned int attempts;
  for (attempts = 0; attempts < MAX_SCATTER_ATTEMPTS; attempts++) {
    // this avoids warp divergence
    if (scatterObj >= N_ENV && sample
            .generatePointInObject(pt, prng, activeRegion)) {
      break;
    } else if (environment[scatterObj]
            .generatePointInObject(pt, prng, activeRegion)) {
      break;
    }
    scatterObj += (scatterObj + 1) % N_SCATTER_OBJS;
  }

  __syncthreads();

  if (attempts > MAX_SCATTER_ATTEMPTS) {
    raise("Unable to generate point in object after the provided number of "
          "attempts");
  }

  return pt;
}

template<DeltaEMode::Type ETYPE>
__global__
void calculateAbsorptionKernel(
        CudaAlgorithmContext * context,
        CudaMersenneTwister * __restrict__ ptrPrng,
        const CudaCSGObject * __restrict__ ptrSample,
        const CudaCSGObject * __restrict__ ptrEnvironment,
        const ICudaBeamProfile * const* __restrict__ ptrBeamProfile,
        const CudaBoundingBox * __restrict__ ptrActiveRegion,
        const CudaBoundingBox * __restrict__ ptrScatterBounds,
        const CudaDetectorBinProvider<ETYPE> * __restrict__ eProviderPtr,
        const CudaMCGeometryBuffer * __restrict__ geometryBuffer,
        double * __restrict__ output) {

  if (context->isCancelled()) {
    raise("Cancelling!");
  }

  const unsigned int eventId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int binId = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int histId = blockIdx.z * blockDim.z + threadIdx.z;

  if (eventId >= N_EVENTS) return;

  // dereference what is needed
  const ICudaBeamProfile &beamProfile = **ptrBeamProfile;
  const CudaBoundingBox &scatterBounds = *ptrScatterBounds;
  const CudaBoundingBox &activeRegion = *ptrActiveRegion;
  const CudaMersenneTwister &prng = *ptrPrng;
  const CudaCSGObject &sample = *ptrSample;

  const CudaDetectorBin &DetectorBin =
            eProviderPtr->getDetectorBin(binId, histId);

  CudaVector<double> distances = geometryBuffer->getDistanceBuffer();
  CudaVector<CudaV3D> points = geometryBuffer->getPointsBuffer();
  CudaVector<CudaLink> links = geometryBuffer->getLinksBuffer();
  CudaVector<CudaIntersectionPoint> surfPoints =
            geometryBuffer->getSurfacePointsBuffer();

  double factor;
  unsigned int attempts;
  for (attempts = 1; attempts <= MAX_SCATTER_ATTEMPTS; attempts++) {
    const CudaV3D &startPos = beamProfile.generatePoint(prng, scatterBounds)
            .startPos;
    const CudaV3D &scatterPos = generatePoint(sample, ptrEnvironment, prng,
                                              activeRegion, eventId);

    links.clear();
    surfPoints.clear();
    distances.clear();
    points.clear();

    // Generate track between points
    CudaV3D normPoint = startPos - scatterPos;
    normPoint.normalize();

    CudaTrack beforeScatter(scatterPos, normPoint, links, surfPoints);

    // intercept the surface
    int nlinks = sample.interceptSurface(beforeScatter, distances, points);
    for (unsigned int i = 0; i < N_ENV; i++) {
      ptrEnvironment[i].interceptSurface(beforeScatter, distances, points);
    }

    __syncthreads();

    if (nlinks < 1) {
      continue;
    }
    __syncthreads();

    factor = calculateAttenuation(beforeScatter, DetectorBin.wavelengthIn);

    links.clear();
    surfPoints.clear();
    distances.clear();
    points.clear();

    // Generate track between points
    normPoint = DetectorBin.detectorPos - scatterPos;
    normPoint.normalize();

    CudaTrack afterScatter(scatterPos, normPoint, links, surfPoints);

    // intercept the surface
    sample.interceptSurface(afterScatter, distances, points);
    for (unsigned int i = 0; i < N_ENV; i++) {
      ptrEnvironment[i].interceptSurface(afterScatter, distances, points);
    }

    factor *= calculateAttenuation(afterScatter, DetectorBin.wavelengthOut);

    break;
  }

  if (attempts >= MAX_SCATTER_ATTEMPTS) {
    raise("Unable to generate valid track through sample interaction volume "
          "Increase the maximum threshold or if this does not help then please "
          "check the defined shape.");
  }

  // Perform reduction
  factor = CudaReduce::warpReduce(factor);

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(output + binId + (histId * N_BINS), factor);
  }

  __syncthreads();
  if (eventId == (N_EVENTS - 1)) {
    context->updateProgress();
  }
}

template<DeltaEMode::Type etype>
void CudaMCAbsorptionStrategy<etype>::calculate(
        const cudaStream_t &algorithmStream) const {

  calculateAbsorptionKernel<etype>
          <<<m_gridDim, m_blockSize, 0, algorithmStream>>>(md_context,
                                                           md_cudaRng,
                                                           md_sample,
                                                           md_environment,
                                                           md_beamProfile,
                                                           md_activeRegion,
                                                           md_scatterBounds,
                                                           md_eProvider,
                                                           md_geometryBuffer,
                                                           md_output);
}

// forward declarations
template class CudaMCAbsorptionStrategy<DeltaEMode::Direct>;
template class CudaMCAbsorptionStrategy<DeltaEMode::Indirect>;
template class CudaMCAbsorptionStrategy<DeltaEMode::Elastic>;

}
}
