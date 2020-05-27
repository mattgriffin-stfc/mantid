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

class CudaV3D;
struct CudaIntersectionPoint;
struct CudaLink;
template<typename T>
class CudaVector;

/**
 * Geometry buffer, abstracts the instantiation and allocation of geometric
 * buffers needed in the CudaMonteCarloAbsorption algorithm kernel.
 */
class CudaMCGeometryBuffer {
public:
  /**
   * Constructor for CudaMCGeometryBuffer
   * @param elements  number of elements in each buffer
   * @param warpSize  the device's warp size
   * @param cores     the number of cores in a single sm
   * @param sms       the number of streaming multiprocessors on a device
   */
  __host__
  CudaMCGeometryBuffer(const unsigned int elements,
                       const unsigned int warpSize,
                       const unsigned int cores,
                       const unsigned int sms);

  /**
   * @return the distance buffer for this thread
   */
  __device__
  CudaVector<double> getDistanceBuffer() const;

  /**
   * @return the points buffer for this thread
   */
  __device__
  CudaVector<CudaV3D> getPointsBuffer() const;

  /**
   * @return the links buffer for this thread
   */
  __device__
  CudaVector<CudaLink> getLinksBuffer() const;

  /**
   * @return the surface points buffer for this thread
   */
  __device__
  CudaVector<CudaIntersectionPoint> getSurfacePointsBuffer() const;

  /**
   * Deconstructor for CudaMCGeometryBuffer, tidy up the vectors
   */
  __host__
  ~CudaMCGeometryBuffer();

private:
  __device__
  unsigned int getMemoryOffset() const;

  CudaV3D * md_points;
  double * md_distances;
  CudaIntersectionPoint * md_surfPoints;
  CudaLink * md_links;
};

}
}
