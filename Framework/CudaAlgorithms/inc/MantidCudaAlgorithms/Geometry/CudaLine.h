// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cuda_runtime.h>

namespace thrust {
template<typename T>
class complex;
}

namespace Mantid {
namespace CudaAlgorithms {

class CudaQuadratic;
class CudaCylinder;
class CudaPlane;
class CudaSphere;
class CudaV3D;

template<typename T>
class CudaVector;

/**
 * The CudaLine class (cut down, CUDA equivalent for Geometry::Line)
 *
 * Constructable on device, usable on device.
 *
 * Impliments the line \f[ r=\vec{O} + \lambda \vec{n} \f]
 */
class CudaLine {
public:
  /**
   * Constructor for CudaLine
   */
  __device__
  CudaLine(const CudaV3D &, const CudaV3D &);

  /**
   * Gets the point O+lam*N
   * @param lambda the distance
   * @return the point at the given wavelength distance
   */
  __device__
  CudaV3D getPoint(const double lambda) const;

  /**
   * Intersect this line with the given surface
   * @param PntOut vector to store the intersection points in
   * @param surf   to intersect this line with
   */
  __device__
  void intersect(CudaVector<CudaV3D> &PntOut, const CudaQuadratic &surf) const;
  __device__
  void intersect(CudaVector<CudaV3D> &PntOut, const CudaCylinder &surf) const;
  __device__
  void intersect(CudaVector<CudaV3D> &PntOut, const CudaPlane &surf) const;
  __device__
  void intersect(CudaVector<CudaV3D> &PntOut, const CudaSphere &surf) const;

  /// returns the origin
  __inline__ __device__
  const CudaV3D &getOrigin() const { return Origin; }
  /// returns the direction
  __inline__ __device__
  const CudaV3D &getDirect() const { return Direct; }

private:
  /// origin of the line
  const CudaV3D &Origin;
  /// direction of the line
  const CudaV3D &Direct;

  /**
   * Calculate intersection points given the solutions to a quadratic equations.
   * @param ix number of solutions
   * @param d1 first solution
   * @param d2 second solution
   * @param PntOut vector of intersection points
   */
  __device__
  void lambdaPair(const int ix,
                  const thrust::complex<double> &d1,
                  const thrust::complex<double> &d2,
                  CudaVector<CudaV3D> &PntOut) const;
};

}
}
