// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidAPI/Algorithm.h"
#include "MantidKernel/DeltaEMode.h"

namespace Mantid {

using namespace Kernel;

namespace CudaAlgorithms {

/**
  Calculates attenuation due to absorption and scattering in a sample +
  its environment using a Monte Carlo algorithm.
*/
class DLLExport CudaMonteCarloAbsorption : public API::Algorithm {
public:
  /// Algorithm's name
  const std::string name() const override { return "CudaMonteCarloAbsorption"; }
  /// Algorithm's version
  int version() const override { return 1; }
  const std::vector<std::string> seeAlso() const override {
    return {"MonteCarloAbsorption"};
  }
  /// Algorithm's category for identification
  const std::string category() const override {
    return "CorrectionFunctions\\AbsorptionCorrections";
  }
  /// Summary of algorithms purpose
  const std::string summary() const override {
    return "Calculates attenuation due to absorption and scattering in a "
           "sample & its environment using a CUDA accelerated Monte Carlo "
           "approach.";
  }

private:
  void init() override;
  void exec() override;

  template<DeltaEMode::Type etype>
  API::MatrixWorkspace_uptr doSimulation(const API::MatrixWorkspace &inputWS,
                                         const unsigned int nevents,
                                         const int seed,
                                         const unsigned int maxScatterAttempts,
                                         const unsigned int blockSize,
                                         const unsigned int geometryBuffer);

  API::MatrixWorkspace_uptr
  createOutputWorkspace(const API::MatrixWorkspace &inputWS) const;
};

}
}
