// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Kernel/CudaMersenneTwister.h"

#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include "MantidCudaAlgorithms/CudaGuard.h"
#include "MantidCudaAlgorithms/CurandGuard.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"


namespace Mantid {
namespace CudaAlgorithms {

/**
 * The maximum number of states curandMakeMTGP32KernelState can initialize at
 * once.
 */
constexpr int MAX_INIT_STATES = 200;
constexpr size_t rngParamsSize = sizeof(mtgp32_kernel_params);

__constant__ unsigned int MT_CORES;

__host__
CudaMersenneTwister::CudaMersenneTwister(const int seed,
                                         const unsigned int cores,
                                         const unsigned int multiprocessors) {

  CUDA_GUARD(cudaMemcpyToSymbol(MT_CORES, &cores, sizeof(int)));

  const int rngStates = multiprocessors * cores;

  const size_t rngStatesSize = sizeof(curandStateMtgp32) * rngStates;

  CUDA_GUARD(cudaMalloc(&curandStates, rngStatesSize));
  CUDA_GUARD(cudaMalloc(&d_rngParams, rngParamsSize));

  // create MTGP32 params from a predefined set
  CURAND_GUARD(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213,
                                         d_rngParams));

  // can only initialize 200 states at a time due to the pregenerated parameter
  // sets.
  for (int i = 0; i < rngStates; i += MAX_INIT_STATES) {
    int initStates = min(rngStates - i, MAX_INIT_STATES);
    CURAND_GUARD(curandMakeMTGP32KernelState(curandStates + i,
                                             mtgp32dc_params_fast_11213,
                                             d_rngParams, initStates,
                                             seed + i));
  }
}

__host__
CudaMersenneTwister::~CudaMersenneTwister() {
  CUDA_SOFT_GUARD(cudaFree(curandStates));
  CUDA_SOFT_GUARD(cudaFree(d_rngParams));
}

__device__
curandStateMtgp32 * CudaMersenneTwister::getState() const {
  return curandStates + warpId() + smId() * MT_CORES;
}

__device__
double CudaMersenneTwister::nextValue() const {
   return curand_uniform_double(getState());
}

}
}
