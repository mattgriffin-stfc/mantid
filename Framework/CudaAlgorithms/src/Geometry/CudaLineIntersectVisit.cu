// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h"

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"
#include "MantidCudaAlgorithms/CudaVector.h"
#include "MantidCudaAlgorithms/Geometry/CudaCone.h"
#include "MantidCudaAlgorithms/Geometry/CudaQuadratic.h"
#include "MantidCudaAlgorithms/Geometry/CudaLine.h"
#include "MantidCudaAlgorithms/CudaAlgorithmContext.h"

namespace Mantid {
namespace CudaAlgorithms {

__device__
CudaLineIntersectVisit::CudaLineIntersectVisit(const CudaLine &line,
                                               CudaVector<double> &dOut,
                                               CudaVector<CudaV3D> &ptOut)
    : DOut(dOut), PtOut(ptOut), ATrack(line) {}


__device__
void CudaLineIntersectVisit::Accept(const CudaQuadratic &Surf) {
  ATrack.intersect(PtOut, Surf);
}

__device__
void CudaLineIntersectVisit::Accept(const CudaPlane &Surf) {
  ATrack.intersect(PtOut, Surf);
}

__device__
void CudaLineIntersectVisit::Accept(const CudaCone &Surf) {
  ATrack.intersect(PtOut, Surf);
}

__device__
void CudaLineIntersectVisit::Accept(const CudaCylinder &Surf) {
  ATrack.intersect(PtOut, Surf);
}

__device__
void CudaLineIntersectVisit::Accept(const CudaSphere &Surf) {
  ATrack.intersect(PtOut, Surf);
}

__device__
const CudaVector<double> &CudaLineIntersectVisit::procTrack() {
  // Calculate the distances to the points

  const CudaV3D &temp = ATrack.getOrigin();
  for(unsigned int i = 0; i < PtOut.size(); i++) {
    double d = temp.distance(PtOut[i]);
    DOut.emplace_back(d);
  }

  __syncthreads();
  return DOut;
}

}
}
