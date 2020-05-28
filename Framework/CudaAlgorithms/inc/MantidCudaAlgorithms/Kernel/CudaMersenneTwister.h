// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +

#include <cuda_runtime.h>

class curandStateMtgp32;
class mtgp32_kernel_params;

namespace Mantid {
namespace CudaAlgorithms {

/**
 * The CudaMersenneTwister class, CUDA equivalent of Kernel::MersenneTwister.
 * PRNG which wraps cuRAND's MTGP32 (GPU optimized Mersenne Twister) generator.
 * Creates a pool of curand states equalling the number of cores on a device so
 * that every warp can accesse a state at once. This minimizes the number of
 * states needed for a kernel.
 */
class CudaMersenneTwister {
public:
  /**
   * Constructor for CudaMersenneTwister, creates the curand states for each
   * core.
   * @param seed  the seed to use for  this PRNG
   * @param cores the number of states to create (usually equal to the number of
   *              cores on the GPU)
   */
  __host__
  CudaMersenneTwister(const int seed, const unsigned int cores,
                      const unsigned int multiprocessors);

  /**
   * Destructor for CudaMersenneTwister, clears the curand states.
   */
  __host__
  ~CudaMersenneTwister();

  /**
   * @return the next value in the PRNG sequence
   */
  __device__
  double nextValue() const;

private:
  curandStateMtgp32 * curandStates;
  mtgp32_kernel_params * d_rngParams;

  /**
   * @return a curand state for this cuda core
   */
  __device__
  curandStateMtgp32 * getState() const;
};

}
}
