// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"

#include <stdio.h>

#include "MantidCudaAlgorithms/CudaGuard.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__ int EXCEPTION_LOCK = 0;

constexpr size_t PROGRESS_SIZE = sizeof(int);
constexpr size_t CANCEL_SIZE   = sizeof(bool);

__host__
CudaAlgorithmContext::CudaAlgorithmContext() {
  CUDA_GUARD(cudaStreamCreate(&m_ctrlStream));

  CUDA_GUARD(cudaMalloc(&md_progress, PROGRESS_SIZE));
  CUDA_GUARD(cudaMalloc(&md_cancel, CANCEL_SIZE));
}

__host__
CudaAlgorithmContext::~CudaAlgorithmContext() {
  CUDA_SOFT_GUARD(cudaStreamDestroy(m_ctrlStream));

  CUDA_SOFT_GUARD(cudaFree(md_progress));
  CUDA_SOFT_GUARD(cudaFree(md_cancel));
}

__device__
void CudaAlgorithmContext::updateProgress() {
  atomicAdd(md_progress, 1);
}

__host__
int CudaAlgorithmContext::getProgress() const {
  int currProg;
  CUDA_GUARD(cudaMemcpyAsync(&currProg, md_progress,
                             PROGRESS_SIZE,
                             cudaMemcpyDeviceToHost,
                             m_ctrlStream));
  CUDA_GUARD(cudaStreamSynchronize(m_ctrlStream));

  return currProg;
}

__device__
bool CudaAlgorithmContext::isCancelled() const {
  return *md_cancel;
}

__host__
void CudaAlgorithmContext::cancel() {
  bool cancel = true;

  CUDA_GUARD(cudaMemcpyAsync(md_cancel,
                             &cancel,
                             CANCEL_SIZE,
                             cudaMemcpyHostToDevice,
                             m_ctrlStream));
  CUDA_GUARD(cudaStreamSynchronize(m_ctrlStream));
}

__device__
void raise(const char * msg) {
  // Create a lock so that the error message is printed only once
  // Don't have to unlock as the device will corrupt
  if (!atomicCAS(&EXCEPTION_LOCK, 0, 1)) {
      printf("[Application Runtime Error] %s\n\n", msg);

      asm("trap;"); // kill kernel with error
  }
}

}
}
