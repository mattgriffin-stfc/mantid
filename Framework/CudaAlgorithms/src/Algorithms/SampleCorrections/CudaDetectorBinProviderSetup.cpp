// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProviderSetup.h"

#include <cstring>
#include <vector>

#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/SpectrumInfo.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProvider.h"
#include "MantidCudaAlgorithms/CudaGuard.h"
#include "MantidKernel/PhysicalConstants.h"
#include "MantidGeometry/IDetector.h"


namespace Mantid {
namespace CudaAlgorithms {

/// Energy (meV) to wavelength (angstroms)
inline double toWavelength(double energy) {
  static const double factor =
      1e10 * PhysicalConstants::h /
      sqrt(2.0 * PhysicalConstants::NeutronMass * PhysicalConstants::meV);
  return factor / sqrt(energy);
}

template<DeltaEMode::Type etype>
void CudaDetectorBinProviderSetup<etype>::setBinLambdas(
        const cudaTextureDesc &td,
        const cudaChannelFormatDesc &fd,
        const API::MatrixWorkspace &instrumentWS,
        const unsigned int nbins,
        const unsigned int nhists,
        const size_t &textureHeight,
        const cudaStream_t &cudaStream) {

  /*
   * Allocate pitched array
   */
  const size_t binRowSize = sizeof(double) * nbins;
  const size_t totalBinSize = nhists * binRowSize;

  CUDA_GUARD(cudaMallocHost(&h_lambdaNotFixed, totalBinSize));

  for (size_t i = 0; i < nhists; i++) {
    const std::vector<double> &vect = instrumentWS.points(i).rawData();
    std::copy(vect.begin(), vect.end(), h_lambdaNotFixed + (nbins * i));
  }

  size_t pitch;
  CUDA_GUARD(cudaMallocPitch(&d_lambdaNotFixed, &pitch,  binRowSize, nhists));
  CUDA_GUARD(cudaMemcpy2DAsync(d_lambdaNotFixed, pitch, h_lambdaNotFixed,
                               binRowSize, binRowSize, nhists,
                               cudaMemcpyHostToDevice, cudaStream));

  /*
   * Convert to textures
   */
  const size_t textures = (nhists + textureHeight - 1) / textureHeight;
  const size_t textureSize = sizeof(cudaTextureObject_t) * textures;

  CUDA_GUARD(cudaMallocHost(&h_binLambdaTextures, textureSize));

  for (size_t i = 0; i < textures; i++) {
    size_t height = std::min(nhists - textureHeight * i, textureHeight);

    cudaResourceDesc lamdaNotFixedResDesc;
    memset(&lamdaNotFixedResDesc, 0, sizeof(lamdaNotFixedResDesc));
    lamdaNotFixedResDesc.resType = cudaResourceTypePitch2D;
    lamdaNotFixedResDesc.res.pitch2D.devPtr = d_lambdaNotFixed +
            (i * textureHeight * pitch/sizeof(double));
    lamdaNotFixedResDesc.res.pitch2D.width = nbins;
    lamdaNotFixedResDesc.res.pitch2D.height = height;
    lamdaNotFixedResDesc.res.pitch2D.pitchInBytes = pitch;
    lamdaNotFixedResDesc.res.pitch2D.desc = fd;

    CUDA_GUARD(cudaCreateTextureObject(h_binLambdaTextures + i,
                                       &lamdaNotFixedResDesc, &td, NULL));
  }

  CUDA_GUARD(cudaMalloc(&d_binLambdaTextures, textureSize));
  CUDA_GUARD(cudaMemcpyAsync(d_binLambdaTextures, h_binLambdaTextures,
                             textureSize, cudaMemcpyHostToDevice, cudaStream));
}

template<DeltaEMode::Type etype>
void CudaDetectorBinProviderSetup<etype>::setHistLambdas(
        const cudaTextureDesc &td,
        const cudaChannelFormatDesc &fd,
        const API::MatrixWorkspace &instrumentWS,
        const unsigned int nhists,
        const cudaStream_t &cudaStream) {

  const auto &spectrumInfo = instrumentWS.spectrumInfo();

  cudaChannelFormatDesc fd128;
  memset(&fd128, 0, sizeof(fd128));
  fd128.f = cudaChannelFormatKindUnsigned;
  fd128.x = 32;
  fd128.y = 32;
  fd128.z = 32;
  fd128.w = 32;

  size_t double2Size = sizeof(double2) * nhists;
  size_t doubleSize = sizeof(double) * nhists;

  CUDA_GUARD(cudaMallocHost(&h_detectorPositionsXY, double2Size));
  CUDA_GUARD(cudaMalloc(&d_detectorPositionsXY, double2Size));

  if (instrumentWS.getEMode() == DeltaEMode::Indirect) {
    CUDA_GUARD(cudaMallocHost(&h_detectorPositionsZW, double2Size));
    CUDA_GUARD(cudaMalloc(&d_detectorPositionsZW, double2Size));
  } else {
    CUDA_GUARD(cudaMallocHost(&h_detectorPositionsZ,  doubleSize));
    CUDA_GUARD(cudaMalloc(&d_detectorPositionsZ, doubleSize));
  }

  for (size_t i = 0; i < nhists; ++i) {
    const V3D &position = spectrumInfo.position(i);
    h_detectorPositionsXY[i] = {position.X(), position.Y()};

    if (instrumentWS.getEMode() == DeltaEMode::Indirect) {
        h_detectorPositionsZW[i] = {position.Z(), toWavelength(
                instrumentWS.getEFixed(spectrumInfo.detector(i).getID()))};
    } else {
        h_detectorPositionsZ[i] = position.Z();
    }
  }

  CUDA_GUARD(cudaMemcpyAsync(d_detectorPositionsXY, h_detectorPositionsXY,
                             double2Size, cudaMemcpyHostToDevice, cudaStream));

  struct cudaResourceDesc detectorXYDesc;
  memset(&detectorXYDesc, 0, sizeof(detectorXYDesc));
  detectorXYDesc.resType = cudaResourceTypeLinear;
  detectorXYDesc.res.linear.sizeInBytes = double2Size;
  detectorXYDesc.res.linear.devPtr = d_detectorPositionsXY;
  detectorXYDesc.res.linear.desc = fd128;

  CUDA_GUARD(cudaCreateTextureObject(&detectorPositionsXYTexture,
                                     &detectorXYDesc, &td, NULL));


  struct cudaResourceDesc lamdaFixedResDesc;
  memset(&lamdaFixedResDesc, 0, sizeof(lamdaFixedResDesc));
  lamdaFixedResDesc.resType = cudaResourceTypeLinear;

  if (instrumentWS.getEMode() == DeltaEMode::Indirect) {
    CUDA_GUARD(cudaMemcpyAsync(d_detectorPositionsZW, h_detectorPositionsZW,
                               double2Size, cudaMemcpyHostToDevice,
                               cudaStream));

    lamdaFixedResDesc.res.linear.sizeInBytes = double2Size;
    lamdaFixedResDesc.res.linear.devPtr = d_detectorPositionsZW;
    lamdaFixedResDesc.res.linear.desc = fd128;
  } else {
    CUDA_GUARD(cudaMemcpyAsync(d_detectorPositionsZ, h_detectorPositionsZ,
                               doubleSize, cudaMemcpyHostToDevice, cudaStream));

    lamdaFixedResDesc.res.linear.sizeInBytes = doubleSize;
    lamdaFixedResDesc.res.linear.devPtr = h_detectorPositionsZ;
    lamdaFixedResDesc.res.linear.desc = fd;
  }

  CUDA_GUARD(cudaCreateTextureObject(&detectorPositionsZWTexture,
                                     &lamdaFixedResDesc, &td, NULL));
}

template<DeltaEMode::Type etype>
CudaDetectorBinProviderSetup<etype>::CudaDetectorBinProviderSetup(
        const API::MatrixWorkspace &instrumentWS,
        const unsigned int nbins,
        const unsigned int nhists,
        const cudaDeviceProp &deviceProperties,
        const cudaStream_t &cudaStream) {

  const unsigned int maxTextureWidth = deviceProperties.maxTexture2DLinear[0];
  const unsigned int maxTextureHeight = deviceProperties.maxTexture2DLinear[1];

  if (nbins > maxTextureWidth) {
      std::ostringstream error;
      error << "Exceeded number of bins per hist supported by device: "
            << nbins << "/" << maxTextureWidth
            << std::endl;
       throw std::runtime_error(error.str());
  }

  cudaTextureDesc td;
  memset(&td, 0, sizeof(td));
  td.normalizedCoords = 0;
  td.addressMode[0] = cudaAddressModeClamp;
  td.addressMode[1] = cudaAddressModeClamp;
  td.readMode = cudaReadModeElementType;

  cudaChannelFormatDesc fd;
  memset(&fd, 0, sizeof(fd));
  fd.f = cudaChannelFormatKindUnsigned;
  fd.x = 32;
  fd.y = 32;

  setBinLambdas(td, fd, instrumentWS, nbins, nhists, maxTextureHeight,
                cudaStream);
  setHistLambdas(td, fd, instrumentWS, nhists, cudaStream);

  double experimentWavelength = 0;

  if (etype == DeltaEMode::Direct) {
    experimentWavelength = toWavelength(instrumentWS.getEFixed());
  }

  CudaDetectorBinProvider<etype> eProvider(d_binLambdaTextures,
                                           detectorPositionsXYTexture,
                                           detectorPositionsZWTexture,
                                           experimentWavelength,
                                           maxTextureHeight);

  const size_t providerSize = sizeof(CudaDetectorBinProvider<etype>);

  CUDA_GUARD(cudaMalloc(&m_eProvider, providerSize));
  CUDA_GUARD(cudaMemcpy(m_eProvider, &eProvider, providerSize,
                        cudaMemcpyHostToDevice));
}

template<DeltaEMode::Type etype>
CudaDetectorBinProviderSetup<etype>::~CudaDetectorBinProviderSetup() {
  CUDA_SOFT_GUARD(cudaFreeHost(h_lambdaNotFixed));
  CUDA_SOFT_GUARD(cudaFree(d_lambdaNotFixed));
  CUDA_SOFT_GUARD(cudaFreeHost(h_binLambdaTextures));
  CUDA_SOFT_GUARD(cudaFree(d_binLambdaTextures));
  CUDA_SOFT_GUARD(cudaFreeHost(h_detectorPositionsXY));
  CUDA_SOFT_GUARD(cudaFree(d_detectorPositionsXY));

  if (etype == DeltaEMode::Indirect) {
    CUDA_SOFT_GUARD(cudaFreeHost(h_detectorPositionsZW));
    CUDA_SOFT_GUARD(cudaFree(d_detectorPositionsZW));
  } else {
    CUDA_SOFT_GUARD(cudaFreeHost(h_detectorPositionsZ));
    CUDA_SOFT_GUARD(cudaFree(d_detectorPositionsZ));
  }

  CUDA_SOFT_GUARD(cudaFree(m_eProvider));
}

// forward declarations
template class CudaDetectorBinProviderSetup<DeltaEMode::Direct>;
template class CudaDetectorBinProviderSetup<DeltaEMode::Indirect>;
template class CudaDetectorBinProviderSetup<DeltaEMode::Elastic>;

}
}
