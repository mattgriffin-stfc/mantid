// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2014 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "DllOption.h"

#include <map>
#include <string>

namespace MantidQt {
namespace MantidWidgets {

class EXPORT_OPT_MANTIDQT_INSTRUMENTVIEW BaseCustomInstrumentModel {

public:
  BaseCustomInstrumentModel();
  ~BaseCustomInstrumentModel(){};
  virtual void loadEmptyInstrument();
  virtual std::pair<int, std::string> loadData(const std::string &name);
  void setCurrentRun(int &run) { m_currentRun = run; };
  int getCurrentRun() { return m_currentRun; };
  void rename();
  void remove();
  std::string dataFileName();
  int currentRun();
  bool isErrorCode(const int run);
  const std::string getInstrument() { return m_instrumentName; };

protected:
  int m_currentRun;
  std::string m_tmpName;
  std::string m_instrumentName;
  std::string m_wsName;
};

} // namespace MantidWidgets
} // namespace MantidQt
