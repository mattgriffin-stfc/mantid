// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaLine.h"

#include <thrust/complex.h>

#include "MantidCudaAlgorithms/Geometry/CudaCylinder.h"
#include "MantidCudaAlgorithms/Geometry/CudaPlane.h"
#include "MantidCudaAlgorithms/Geometry/CudaQuadratic.h"
#include "MantidCudaAlgorithms/Geometry/CudaSphere.h"

#include "MantidCudaAlgorithms/CudaMathExtensions.h"
#include "MantidCudaAlgorithms/CudaVector.h"


namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaLine::CudaLine(const CudaV3D &O, const CudaV3D &D)
    : Origin(O), Direct(D) {}

__device__
CudaV3D CudaLine::getPoint(const double lambda) const {
  return Origin + Direct * lambda;
}

__device__
void CudaLine::lambdaPair(const int ix,
                          const thrust::complex<double> &d1,
                          const thrust::complex<double> &d2,
                          CudaVector<CudaV3D> &PntOut) const {

  // only add if need to
  if (ix >= 1) {
    int currentPoints = 0;

    CudaV3D Ans1;
    if (CudaMathExtensions::isZero(d1.imag())
            && d1.real() >= 0.0) // +ve roots only
    {
      const double lambda = d1.real();
      Ans1 = getPoint(lambda);
      currentPoints = 1;
      PntOut.emplace_back(Ans1);
    }

    if (currentPoints < ix && CudaMathExtensions::isZero(d2.imag())
            && d2.real() >= 0.0) // +ve roots only
    {
      const double lambda = d2.real();
      CudaV3D Ans2 = getPoint(lambda);

      // if there is already a point and its too close then dont add it
      if (!(currentPoints > 0 && CudaMathExtensions::isZero(
                Ans1.distance(Ans2)))) {
        PntOut.emplace_back(Ans2);
      }
    }
  }

  __syncthreads();
}

__device__
void CudaLine::intersect(CudaVector<CudaV3D> &PntOut,
                         const CudaQuadratic &Sur) const {
  const double * BN = Sur.copyBaseEqn();
  const double a(Origin.X()), b(Origin.Y()), c(Origin.Z());
  const double d(Direct.X()), e(Direct.Y()), f(Direct.Z());
  double3 Coef;
  Coef.x = BN[0] * d * d + BN[1] * e * e + BN[2] * f * f + BN[3] * d * e +
           BN[4] * d * f + BN[5] * e * f;
  Coef.y = 2 * BN[0] * a * d + 2 * BN[1] * b * e + 2 * BN[2] * c * f +
           BN[3] * (a * e + b * d) + BN[4] * (a * f + c * d) +
           BN[5] * (b * f + c * e) + BN[6] * d + BN[7] * e + BN[8] * f;
  Coef.z = BN[0] * a * a + BN[1] * b * b + BN[2] * c * c + BN[3] * a * b +
           BN[4] * a * c + BN[5] * b * c + BN[6] * a + BN[7] * b + BN[8] * c +
           BN[9];

  thrust::complex<double> d1;
  thrust::complex<double> d2;

  const int ix = CudaMathExtensions::solveQuadratic(Coef, d1, d2);
  lambdaPair(ix, d1, d2, PntOut);
}

__device__
void CudaLine::intersect(CudaVector<CudaV3D> &PntOut,
                         const CudaPlane &Pln) const {
  const double OdotN = Origin.scalar_prod(Pln.getNormal());
  const double DdotN = Direct.scalar_prod(Pln.getNormal());

  // Plane and line parallel
  if (!CudaMathExtensions::isZero(DdotN)) {
    const double u = (Pln.getDistance() - OdotN) / DdotN;
    if (u > 0) {
      CudaV3D ved = getPoint(u);
      PntOut.emplace_back(ved);
    }
  }

  __syncthreads();
}

__device__
void CudaLine::intersect(CudaVector<CudaV3D> &PntOut,
                         const CudaCylinder &Cyl) const {
  const CudaV3D &Cent = Cyl.getCentre();
  const CudaV3D &Ax = Origin - Cent;
  const CudaV3D &N = Cyl.getNormal();
  const double R = Cyl.getRadius();
  const double vDn = N.scalar_prod(Direct);
  const double vDA = N.scalar_prod(Ax);

  // First solve the equation of intersection
  const double3 &C = {
      fma(-vDn, vDn, 1.0),
      2.0 * fma(-vDA, vDn, Ax.scalar_prod(Direct)),
      fma(-R, R, fma(-vDA, vDA, Ax.scalar_prod(Ax)))
  };

  thrust::complex<double> d1;
  thrust::complex<double> d2;
  const int ix = CudaMathExtensions::solveQuadratic(C, d1, d2);

  // This takes the centre displacement into account:
  lambdaPair(ix, d1, d2, PntOut);
}

__device__
void CudaLine::intersect(CudaVector<CudaV3D> &PntOut,
                         const CudaSphere &Sph) const {

  // Nasty stripping of useful stuff from sphere
  const CudaV3D &Ax = Origin - Sph.getCentre();
  const double R = Sph.getRadius();

  // First solve the equation of intersection
  const double3 &C = {
      1,
      2.0 * Ax.scalar_prod(Direct),
      fma(-R, R, Ax.scalar_prod(Ax))
  };

  thrust::complex<double> d1;
  thrust::complex<double> d2;
  const int ix = CudaMathExtensions::solveQuadratic(C, d1, d2);

  // This takes the centre displacement into account:
  lambdaPair(ix, d1, d2, PntOut);
}

}
}
