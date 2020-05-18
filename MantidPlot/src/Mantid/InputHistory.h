// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2007 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidKernel/SingletonHolder.h"
#include <memory>

#include <QList>
#include <QMap>
#include <QString>

namespace Mantid {
namespace API {
class IAlgorithm;
using IAlgorithm_sptr = std::shared_ptr<IAlgorithm>;
} // namespace API
} // namespace Mantid

struct PropertyData {
  PropertyData(const QString &nm, const QString &vl) : name(nm), value(vl) {}
  QString name;
  QString value;
};

/** @class InputHistory

 Keeps history of Mantid-related user input, such as algorithm parameters, etc.

 @author Roman Tolchenov, Tessella Support Services plc
 @date 15/10/2008
 */

class InputHistoryImpl {
public:
  InputHistoryImpl(const InputHistoryImpl &) = delete;
  InputHistoryImpl &operator=(const InputHistoryImpl &) = delete;
  void updateAlgorithm(const Mantid::API::IAlgorithm_sptr &alg);
  /// The name:value map of non-default properties with which algorithm algName
  /// was called last time.
  QMap<QString, QString> algorithmProperties(const QString &algName);
  /// Returns the value of property propNameif it has been recorded for
  /// algorithm algName.
  QString algorithmProperty(const QString &algName, const QString &propName);
  /// Replaces the value of a recorded property.
  void updateAlgorithmProperty(const QString &algName, const QString &propName,
                               const QString &propValue);
  /// Saves the properties.
  void save();

  /// Returns the directory name from a full file path.
  static QString getDirectoryFromFilePath(const QString &filePath);
  /// Returns the short file name (without extension) from a full file path.
  static QString getNameOnlyFromFilePath(const QString &filePath);

private:
  friend struct Mantid::Kernel::CreateUsingNew<InputHistoryImpl>;

  /// Private Constructor
  InputHistoryImpl();
  /// Private Destructor
  virtual ~InputHistoryImpl() = default;

  /// For debugging
  // cppcheck-suppress unusedPrivateFunction
  void printAll();

  /// Keeps algorithm parameters.
  QMap<QString, QList<PropertyData>> m_history;
};

using InputHistory = Mantid::Kernel::SingletonHolder<InputHistoryImpl>;
