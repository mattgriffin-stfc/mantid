// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"

namespace Mantid {
namespace CudaAlgorithms {

/**
 * The CudaDetectorBin struct, contains information for a given event in the
 * CudaMonteCarloAbsorption algorithm.
 *
 * Should be returned from CudaDetectorBinProvider within a kernel.
 */
struct CudaDetectorBin {
  CudaV3D detectorPos;
  double wavelengthIn;
  double wavelengthOut;
};

}
}
