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

namespace API {
class MatrixWorkspace;
}

namespace CudaAlgorithms {

template<DeltaEMode::Type>
class CudaDetectorBinProvider;

/**
 * The CudaDetectorBinProviderSetup class
 *
 * Constructable on host, usable on host.
 *
 * Safely builds an instance of CudaDetectorBinProvider for the device.
 */
template<DeltaEMode::Type etype>
class CudaDetectorBinProviderSetup {
public:
  CudaDetectorBinProviderSetup(const API::MatrixWorkspace &instrumentWS,
                               const unsigned int nbins,
                               const unsigned int nhists,
                               const cudaDeviceProp &deviceProperties,
                               const cudaStream_t &cudaStream = 0);

  ~CudaDetectorBinProviderSetup();

  /**
   * @return the constructed CudaLambdaProvider instance
   */
  inline CudaDetectorBinProvider<etype> * getProvider() const {
    return m_eProvider;
  }

private:
  CudaDetectorBinProvider<etype> * m_eProvider;

  /// temporary device/host mem
  double2 * d_detectorPositionsXY;
  double2 * h_detectorPositionsXY;
  double2 * d_detectorPositionsZW;
  double2 * h_detectorPositionsZW;
  double * d_detectorPositionsZ;
  double * h_detectorPositionsZ;

  double * d_lambdaNotFixed;
  double * h_lambdaNotFixed;

  /// temporary textures
  cudaTextureObject_t * h_binLambdaTextures;
  cudaTextureObject_t * d_binLambdaTextures;

  cudaTextureObject_t detectorPositionsXYTexture;
  cudaTextureObject_t detectorPositionsZWTexture;

  void setBinLambdas(const cudaTextureDesc &td,
                     const cudaChannelFormatDesc &fd,
                     const API::MatrixWorkspace &instrumentWS,
                     const unsigned int nbins,
                     const unsigned int nhists,
                     const size_t &textureHeight,
                     const cudaStream_t & cudaStream = 0);

  void setHistLambdas(const cudaTextureDesc &td,
                      const cudaChannelFormatDesc &fd,
                      const API::MatrixWorkspace &instrumentWS,
                      const unsigned int nhists,
                      const cudaStream_t & cudaStream = 0);
};

}
}
