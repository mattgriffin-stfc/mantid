// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MSDFitModel.h"

using namespace Mantid::API;

namespace MantidQt {
namespace CustomInterfaces {
namespace IDA {

std::string MSDFitModel::sequentialFitOutputName() const {
  if (isMultiFit())
    return "MultiMSDFit_" + fitModeToName[getFittingMode()] + "_Results";
  return createOutputName("%1%_MSDFit_" + fitModeToName[getFittingMode()] +
                              "_" + m_fitType + "_s%2%",
                          "_to_", TableDatasetIndex{0});
}

std::string MSDFitModel::simultaneousFitOutputName() const {
  return sequentialFitOutputName();
}

std::string MSDFitModel::singleFitOutputName(TableDatasetIndex index,
                                             WorkspaceIndex spectrum) const {
  return createSingleFitOutputName(
      "%1%_MSDFit_" + fitModeToName[getFittingMode()] + "_s%2%_Results", index,
      spectrum);
}

std::vector<std::string> MSDFitModel::getSpectrumDependentAttributes() const {
  return {};
}

std::string MSDFitModel::getResultXAxisUnit() const { return "Temperature"; }

void MSDFitModel::setFitTypeString(const std::string &fitType) {
  m_fitType = fitType;
}

} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt
