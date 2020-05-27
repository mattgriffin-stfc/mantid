// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/CudaMathExtensions.h"

#include <thrust/complex.h>

namespace Mantid {
namespace CudaAlgorithms {
namespace CudaMathExtensions {

__device__
int solveQuadratic(const double3 &Coef,
                   thrust::complex<double> &d1,
                   thrust::complex<double> &d2) {

  int uniqueSolutions = 0;

  double a, b, c;
  a = Coef.x;
  b = Coef.y;
  c = Coef.z;

  if (isZero(a)) {
    int zeroFactor = !isZero(Coef.y);

    d1 = thrust::complex<double>(zeroFactor * (-Coef.z / Coef.y), 0.0);
    d2 = d1;

    uniqueSolutions = zeroFactor;
  } else {
    // b^2 - 4ac
    double cf = b * b - 4 * a * c;
    if (cf >= 0) /* Real Roots */
    {
      const double rcf = sqrt(cf) * ((Coef.y >= 0) * 2 - 1);
      const double q = -0.5 * (Coef.y + rcf);

      d1 = thrust::complex<double>(q / Coef.x, 0.0);
      d2 = thrust::complex<double>(Coef.z / q, 0.0);
      uniqueSolutions = !isZero(cf) + 1;
    } else {
      const double real = ((Coef.y >= 0) * 2 - 1) * -0.5 * sqrt(-cf);
      thrust::complex<double> CQ(-0.5 * Coef.y, real);
      d1 = CQ / Coef.x;
      d2 = Coef.z / CQ;
      uniqueSolutions = 2;
    }
  }


  __syncthreads();

  return uniqueSolutions;
}

}
}
}
