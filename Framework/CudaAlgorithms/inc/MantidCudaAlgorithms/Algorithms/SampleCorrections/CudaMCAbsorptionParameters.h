// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidKernel/DeltaEMode.h"

namespace Mantid {

using namespace Kernel;

namespace CudaAlgorithms {

class CudaMCGeometryBuffer;
class ICudaBeamProfile;
class CudaBoundingBox;
template<DeltaEMode::Type>
class CudaDetectorBinProvider;
class CudaAlgorithmContext;
class CudaMersenneTwister;
class CudaCSGObject;

template<DeltaEMode::Type etype>
struct CudaMCAbsorptionParameters {
  ICudaBeamProfile ** beamProfile;
  CudaBoundingBox * scatterBounds;
  CudaBoundingBox * activeRegion;
  CudaDetectorBinProvider<etype> * eProvider;
  CudaCSGObject * sample;
  CudaMCGeometryBuffer * geometryBuffer;
  CudaCSGObject * environment;
  size_t nevents;
  unsigned int nenv;
  unsigned int nbins;
  unsigned int nhists;
  size_t blockSize;
  size_t maxScatterPtAttempts;
  CudaMersenneTwister * prng;
  double * output;
  CudaAlgorithmContext * context;
};

}
}
