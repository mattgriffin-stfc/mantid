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
struct CudaDetectorBin;

/**
 * @brief The CudaDetectorBinProvider class
 *
 * Constructable on host, usable on device.
 *
 * Abstraction similar to the EFixedProvider in MonteCarloAbsorption for
 * acessing the in and out wavelengths for a given bin in a given detector
 * depending on the experiments DeltaEMode::Type.
 *
 * Additionally provides the detector X, Y and Z co-ordinates. Implementation is
 * optimized for locality using CUDA texture objects.
 *
 * Should be constructed using CudaLambdaProviderSetup.
 */
template<DeltaEMode::Type etype>
class CudaDetectorBinProvider {
  /// Wavelengths for each bin
  const cudaTextureObject_t * m_binLambdas;
  /// X and Y positions for each detector
  const cudaTextureObject_t m_detectorXY;
  /// Z position for each detector and optionally the detectors wavelength
  const cudaTextureObject_t m_detectorZW;

  /**
   * Get properties for simulated events with the given detector.
   * @param detector detector of the simulated event
   * @return CudaDetectorBin struct describing the properties of simulated
   *         events within the given detector
   */
  __device__
  void setDetectorProperties(CudaDetectorBin &DetectorBin,
                             const unsigned int detector) const;

  /**
   * Get properties for simulated events with the given bin (within detector).
   * @param bin      bin id of the simulated event
   * @param detector detector of the simulated event
   * @return CudaDetectorBin struct describing the properties of simulated
   *         events within the given bin (within detector)
   */
  __device__
  void setBinProperties(CudaDetectorBin &DetectorBin, const unsigned int bin,
                        const unsigned int detector) const;

public:
  /**
   * CudaDetectorBinProvider constructor
   * @param binLambdas wavelengths of the bins within a detector as a 2d texture
   * @param detectorXY detector x and y coordinates as a 1d texture
   * @param detectorZW detector z coordinate and wavelength if required as a 1d
   *        texture
   */
  __host__
  CudaDetectorBinProvider(const cudaTextureObject_t * binLambdas,
                          const cudaTextureObject_t &detectorXY,
                          const cudaTextureObject_t &detectorZW,
                          const double experimentLambda,
                          const unsigned int max2dTextureSize);

  /**
   * @brief Get properties for simulated events with the given bin and detector.
   * For optimum performance within a kernel, ensure that all events for a bin
   * and detector before moving on to the next bin/detector.
   * @param bin      bin id of the simulated event
   * @param detector detector of the simulated event
   * @return CudaDetectorBin struct describing the properties of simulated
   *         events within the given bin and detector
   */
  __device__
  CudaDetectorBin getDetectorBin(const unsigned int bin,
                                 const unsigned int detector) const;
};

}
}
