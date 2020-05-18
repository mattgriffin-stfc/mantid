// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2016 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidAlgorithms/DllConfig.h"
#include <memory>
#include <vector>

namespace Mantid {
namespace Algorithms {

/** MaxentTransform : Abstract base class defining MaxEnt transformations from
  image space to data space and vice-versa
*/
class MANTID_ALGORITHMS_DLL MaxentTransform {
public:
  // Constructor
  MaxentTransform() = default;
  // Destructor
  virtual ~MaxentTransform() = default;
  // Transfoms form image space to data space
  virtual std::vector<double> imageToData(const std::vector<double> &image) = 0;
  // Transforms from data space to image space
  virtual std::vector<double> dataToImage(const std::vector<double> &data) = 0;
};

using MaxentTransform_sptr = std::shared_ptr<MaxentTransform>;

} // namespace Algorithms
} // namespace Mantid
