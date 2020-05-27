// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

namespace Mantid {
namespace CudaAlgorithms {

/**
 * The CudaAlgorithmContext class, provides mechanisms for monitoring a CUDA
 * algorithm.
 *
 * Provides abstractions for interacting with an in-flight CUDA algorithm
 * kernel asynchronously through a control stream. This includes sending
 * cancellation requests and progress feedbacks.
 */
class CudaAlgorithmContext {
public:
  /**
   * Constructor for CudaAlgorithmContext, allocates device memory for
   * md_progress and md_cancel and creates a controls stream.
   */
  __host__
  CudaAlgorithmContext();

  /**
   * Destructor for CudaAlgorithmContext, frees device memory and destroys the
   * control stream.
   */
  __host__
  ~CudaAlgorithmContext();

  /**
   * Update the progress of an algorithm by 1.
   */
  __device__
  void updateProgress();

  /**
   * @return the current algorithms progress as an aggregate value. The value
   *         returned should correspond to the algorithms Progress member.
   */
  __host__
  int getProgress() const;

  /**
   * @return whether the algorithm is cancelled, device code should use this
   *         method to check if the current algorithm should terminate.
   */
  __device__
  bool isCancelled() const;

  /**
   * Cancel the current algorithm from the host.
   */
  __host__
  void cancel();

private:
  /// device pointer for the algorithms progress
  int * md_progress;
  /// device pointer for the algorithms cancel flag
  bool * md_cancel;

  /// the control stream for interacting with the
  cudaStream_t m_ctrlStream;
};

/**
 * Immediately kills the device called from.
 * @param message the error message to display
 */
__device__
void raise(const char * message);

/**
 * @return the current threads streaming multiprocessor id
 */
__inline__ __device__
unsigned int smId() {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

/**
 * @return the current threads warp id
 */
__inline__ __device__
unsigned int warpId() {
  unsigned int warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

/**
 * @return the current threads lane id
 */
__inline__ __device__
unsigned int laneId() {
  unsigned int laneId;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

}
}
