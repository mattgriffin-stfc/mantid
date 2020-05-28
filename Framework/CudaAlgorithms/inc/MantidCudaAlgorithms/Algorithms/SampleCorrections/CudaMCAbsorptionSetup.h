// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaBeamProfileFactory.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaCSGObjectFactory.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProviderSetup.h"
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionParameters.h"


namespace Mantid {

namespace API {
class MatrixWorkspace;
class Sample;
}

namespace Geometry {
class IObject;
class SampleEnvironment;

namespace detail {
class ShapeInfo;
}
}

namespace CudaAlgorithms {
class CudaV3D;
class CudaMersenneTwister;
class CudaShapeInfo;
class CudaSurface;
class CudaRule;
class CudaMCGeometryBuffer;
class CudaMaterial;
class CudaAlgorithmContext;

template<DeltaEMode::Type etype>
class CudaMCAbsorptionSetup {
public:

  CudaMCAbsorptionSetup(const API::MatrixWorkspace &instrumentWS,
                        const CudaAlgorithmContext &context,
                        const unsigned int nhists,
                        const unsigned int nbins,
                        const unsigned int nevents,
                        const unsigned int blockSize,
                        const unsigned int maxScatterPtAttempts,
                        const unsigned int geometryBuffer,
                        const int seed,
                        const cudaDeviceProp &properties,
                        const cudaStream_t &algorithmStream = 0);

  ~CudaMCAbsorptionSetup();

  inline const CudaMCAbsorptionParameters<etype> &getDeviceParams() const {
      return m_deviceParams;
  }

private:
  CudaCSGObject * createEnvironment(const Geometry::SampleEnvironment &env,
                                    const cudaStream_t &algorithmStream = 0);

  CudaCSGObject * createSample(const API::Sample &sample,
                               const cudaStream_t &algorithmStream = 0);

  CudaDetectorBinProviderSetup<etype> eProviderFactory;
  CudaMersenneTwister * mt;

  CudaBoundingBox * md_environmentBounds;

  CudaMCAbsorptionParameters<etype> m_deviceParams;
  CudaMCGeometryBuffer * h_geomBuffer;

  CudaBeamProfileFactory beamFactory;
  CudaCSGObjectFactory objectFactory;
};

}
}
