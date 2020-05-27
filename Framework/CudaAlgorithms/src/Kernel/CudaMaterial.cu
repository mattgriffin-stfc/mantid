// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Kernel/CudaMaterial.h"

namespace Mantid {
namespace CudaAlgorithms {

__constant__ const int FACTOR = -100;

__host__
CudaMaterial::CudaMaterial(const double numberDensity,
                           const double totalScatterXSection,
                           const double linearAbsorpXSectionByWL)
    : m_numberDensity(numberDensity),
      m_totalScatterXSection(totalScatterXSection),
      m_linearAbsorpXSectionByWL(linearAbsorpXSectionByWL) {
}

__device__
double CudaMaterial::totalScatterXSection() const {
  return m_totalScatterXSection;
}

__device__
double CudaMaterial::absorbXSection(const double lambda) const {
  return m_linearAbsorpXSectionByWL * lambda;
}

__device__
double CudaMaterial::attenuation(const double distance,
                                 const double lambda) const {
  return exp(FACTOR * m_numberDensity * (totalScatterXSection()
               + absorbXSection(lambda)) * distance);
}

}
}
