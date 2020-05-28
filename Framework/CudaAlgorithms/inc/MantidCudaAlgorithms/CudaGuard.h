// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <sstream>
#include <iostream>

#include <cuda_runtime.h>

/**
 * Macro for wrapping CUDA calls.
 * Ex: CUDA_GUARD(cudaMalloc(&devicePtr));
 */
#define CUDA_GUARD(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define CUDA_SOFT_GUARD(ans) { cudaAssert((ans), __FILE__, __LINE__, false); }

/**
 * @brief gpuAssert  assert a CUDA runtime invocation result
 *
 * Handles an unknown cudaError_t in the CUDA runtime environment.
 *
 * @param code  the cudaError code that occurred
 * @param file  file in which the error occurred
 * @param line  line number at which the error occurred
 * @param die   whether to throw an exception on error (default: true).
 */
inline void cudaAssert(cudaError_t code, const char* file, int line,
                       bool die = true) {
  if (code != cudaSuccess) {
    const char * errorMessage = cudaGetErrorString(code);

    std::ostringstream error;
    error << "CUDA Error: " << errorMessage
          << " FILE: " << file
          << " LINE: " << line
          << std::endl;

    cudaDeviceReset();
    if (die) {
        throw std::runtime_error(error.str());
    } else {
        std::cerr << error.str();
    }
  }
}

/**
 * Returns the number of cores on the device dependent on the major and minor
 * compute compatiblities/codes.
 * See: https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h.
 * @param major version of the GPU ie X.1
 * @param minor version of the GPU ie 1.X
 * @return number of cores on the device
 */
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}
