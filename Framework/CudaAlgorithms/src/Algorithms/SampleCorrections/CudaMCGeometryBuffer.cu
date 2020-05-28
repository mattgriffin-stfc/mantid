// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCGeometryBuffer.h"

#include "MantidCudaAlgorithms/Geometry/CudaTrack.h"
#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/CudaVector.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"
#include "MantidCudaAlgorithms/CudaGuard.h"

namespace Mantid {
namespace CudaAlgorithms {

__constant__ unsigned int GEOMETRY_BUFFER;
__constant__ unsigned int GB_CORES;

__host__
CudaMCGeometryBuffer::CudaMCGeometryBuffer(const unsigned int elements,
                                           const unsigned int warpSize,
                                           const unsigned int cores,
                                           const unsigned int multiprocessors) {

  CUDA_GUARD(cudaMemcpyToSymbol(GEOMETRY_BUFFER, &elements, sizeof(int)));
  CUDA_GUARD(cudaMemcpyToSymbol(GB_CORES, &cores, sizeof(int)));

  int totalElements = elements * warpSize * cores * multiprocessors;

  CUDA_GUARD(cudaMalloc(&md_points, sizeof(CudaV3D) * totalElements));
  CUDA_GUARD(cudaMalloc(&md_distances, sizeof(double) * totalElements));
  CUDA_GUARD(cudaMalloc(&md_links, sizeof(CudaLink) * totalElements));
  CUDA_GUARD(cudaMalloc(&md_surfPoints,
                        sizeof(CudaIntersectionPoint) * totalElements));
}

__device__
unsigned int CudaMCGeometryBuffer::getMemoryOffset() const {
  return (laneId() + (warpId() * warpSize) + (smId() * GB_CORES * warpSize))
          * GEOMETRY_BUFFER;
}

__device__
CudaVector<double> CudaMCGeometryBuffer::getDistanceBuffer() const {
  return CudaVector<double>(md_distances + getMemoryOffset(), GEOMETRY_BUFFER);
}

__device__
CudaVector<CudaV3D> CudaMCGeometryBuffer::getPointsBuffer() const {
  return CudaVector<CudaV3D>(md_points + getMemoryOffset(), GEOMETRY_BUFFER);
}

__device__
CudaVector<CudaLink> CudaMCGeometryBuffer::getLinksBuffer() const {
  return CudaVector<CudaLink>(md_links + getMemoryOffset(), GEOMETRY_BUFFER);
}

__device__
CudaVector<CudaIntersectionPoint> CudaMCGeometryBuffer::getSurfacePointsBuffer()
    const {
  return CudaVector<CudaIntersectionPoint>(md_surfPoints + getMemoryOffset(),
                                           GEOMETRY_BUFFER);
}

__host__
CudaMCGeometryBuffer::~CudaMCGeometryBuffer() {
  CUDA_SOFT_GUARD(cudaFree(md_points));
  CUDA_SOFT_GUARD(cudaFree(md_distances));
  CUDA_SOFT_GUARD(cudaFree(md_surfPoints));
  CUDA_SOFT_GUARD(cudaFree(md_links));
}

}
}
