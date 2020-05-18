// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory UKRI,
//     NScD Oak Ridge National Laboratory, European Spallation Source
//     & Institut Laue - Langevin
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "Common/DllConfig.h"

namespace MantidQt {
namespace CustomInterfaces {
namespace ISISReflectometry {

class MANTIDQT_ISISREFLECTOMETRY_DLL OptionsDialogPresenterSubscriber {
public:
  virtual void notifyOptionsChanged() const = 0;
};

class MANTIDQT_ISISREFLECTOMETRY_DLL IOptionsDialogPresenter {
public:
  virtual ~IOptionsDialogPresenter() = default;
  virtual void notifySubscribeView() = 0;
  virtual bool getBoolOption(const std::string &optionName) = 0;
  virtual int &getIntOption(const std::string &optionName) = 0;
  virtual void showView() = 0;
  virtual void subscribe(OptionsDialogPresenterSubscriber *notifyee) = 0;
};

} // namespace ISISReflectometry
} // namespace CustomInterfaces
} // namespace MantidQt
