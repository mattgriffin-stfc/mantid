// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProvider.h"

#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBin.h"
#include "MantidCudaAlgorithms/CudaGuard.h"

namespace Mantid {
namespace CudaAlgorithms {

//// Lambda that is constant across entire experiment
__constant__ double EXPERIMENT_LAMBDA;
__constant__ unsigned int MAX_2D_TEXTURE_SIZE;


template<DeltaEMode::Type etype>
__host__
CudaDetectorBinProvider<etype>::CudaDetectorBinProvider(
        const cudaTextureObject_t * binLambdas,
        const cudaTextureObject_t &detectorXY,
        const cudaTextureObject_t &detectorZW,
        const double experimentLambda,
        const unsigned int max2dTextureSize)
    : m_binLambdas(binLambdas),
      m_detectorXY(detectorXY),
      m_detectorZW(detectorZW) {

  CUDA_GUARD(cudaMemcpyToSymbol(EXPERIMENT_LAMBDA, &experimentLambda,
                                sizeof(double)));
  CUDA_GUARD(cudaMemcpyToSymbol(MAX_2D_TEXTURE_SIZE, &max2dTextureSize,
                                sizeof(int)));
}

template<DeltaEMode::Type etype>
__device__
void CudaDetectorBinProvider<etype>::setDetectorProperties(
        CudaDetectorBin &detectorBin,
        const unsigned int detector) const {

  const uint4 detectorXY = tex1Dfetch<uint4>(m_detectorXY, detector);
  detectorBin.detectorPos.setX(__hiloint2double(detectorXY.y, detectorXY.x));
  detectorBin.detectorPos.setY(__hiloint2double(detectorXY.w, detectorXY.z));

  if (etype == DeltaEMode::Indirect) {
    const uint4 detectorZW = tex1Dfetch<uint4>(m_detectorZW, detector);
    detectorBin.detectorPos.setZ(__hiloint2double(detectorZW.y, detectorZW.x));
    detectorBin.wavelengthOut = __hiloint2double(detectorZW.w, detectorZW.z);

  } else {
    const uint2 detectorZ = tex1Dfetch<uint2>(m_detectorZW, detector);
    detectorBin.detectorPos.setZ(__hiloint2double(detectorZ.y, detectorZ.x));
  }
}

template<DeltaEMode::Type etype>
__device__
void CudaDetectorBinProvider<etype>::setBinProperties(
        CudaDetectorBin &detectorBin,
        const unsigned int bin,
        const unsigned int detector) const {

  unsigned short textureId = detector / MAX_2D_TEXTURE_SIZE;
  unsigned short textureWidth = detector % MAX_2D_TEXTURE_SIZE;

  cudaTextureObject_t texture = m_binLambdas[textureId];

  const uint2 binLambda = tex2D<uint2>(texture, bin, textureWidth);

  if (etype == DeltaEMode::Direct) {
    detectorBin.wavelengthIn = EXPERIMENT_LAMBDA;
    detectorBin.wavelengthOut = __hiloint2double(binLambda.y, binLambda.x);

  } else if (etype == DeltaEMode::Indirect) {
    detectorBin.wavelengthIn = __hiloint2double(binLambda.y, binLambda.x);

  } else {
    detectorBin.wavelengthIn = __hiloint2double(binLambda.y, binLambda.x);
    detectorBin.wavelengthOut = detectorBin.wavelengthIn;
  }
}

template<DeltaEMode::Type etype>
__device__
CudaDetectorBin CudaDetectorBinProvider<etype>::getDetectorBin(
        const unsigned int bin,
        const unsigned int detector) const {
  CudaDetectorBin detectorBin;

  setDetectorProperties(detectorBin, detector);
  setBinProperties(detectorBin, bin, detector);

  return detectorBin;
}

template class CudaDetectorBinProvider<DeltaEMode::Direct>;
template class CudaDetectorBinProvider<DeltaEMode::Indirect>;
template class CudaDetectorBinProvider<DeltaEMode::Elastic>;

}
}
