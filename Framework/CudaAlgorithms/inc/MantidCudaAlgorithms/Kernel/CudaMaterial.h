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

/**
 * The CudaMaterial class (cut down, CUDA equivalent for Geometry::Material)
 *
 * Constructable on host and device, usable on device.
 */
class CudaMaterial final {
public:
  /**
   * Construct a material object
   * @param name :: The name of the material
   * @param formula :: The chemical formula
   * @param numberDensity :: Density in atoms / Angstrom^3
   * @param temperature :: The temperature in Kelvin (Default = 300K)
   * @param pressure :: Pressure in kPa (Default: 101.325 kPa)
   */
  __host__
  explicit CudaMaterial(const double numberDensity,
                        const double totalScatterXSection,
                        const double linearAbsorpXSectionByWL);

  /**
   * Get the total scattering cross section following Sears eqn 13.
   *
   * @returns The value of the total scattering cross section.
   */
  __device__
  double totalScatterXSection() const;

  /**
   * Get the absorption cross section for a given wavelength
   * @param lambda :: The wavelength to evaluate the cross section
   * @returns The value of the absoprtion cross section at
   * the given wavelength
   */
  __device__
  double absorbXSection(const double lambda) const;

  /**
   * @param distance Distance (m) travelled
   * @param lambda Wavelength (Angstroms) to compute the attenuation (default =
   * reference lambda)
   * @return The dimensionless attenuation coefficient
   */
  __device__
  double attenuation(const double distance, const double lambda) const;

private:
  /// Number density in atoms per A^-3
  const double m_numberDensity;
  const double m_totalScatterXSection;
  const double m_linearAbsorpXSectionByWL;
};

}
}
