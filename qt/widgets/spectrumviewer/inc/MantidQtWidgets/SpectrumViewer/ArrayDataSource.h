// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2012 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidQtWidgets/SpectrumViewer/DataArray.h"
#include "MantidQtWidgets/SpectrumViewer/SpectrumDataSource.h"
#include <cstddef>

/**
    @class ArrayDataSource

    This class provides a wrapper around a simple 2-D array of doubles
    stored in row-major order in a 1-D array, so that the array can be
    viewed using the SpectrumView data viewer.

    @author Dennis Mikkelson
    @date   2012-05-14
 */

namespace MantidQt {
namespace SpectrumView {

class EXPORT_OPT_MANTIDQT_SPECTRUMVIEWER ArrayDataSource
    : public SpectrumDataSource {
public:
  /// Construct a DataSource object based on the specified array of floats
  ArrayDataSource(double total_xmin, double total_xmax, double total_ymin,
                  double total_ymax, size_t total_rows, size_t total_cols,
                  std::vector<float> data);

  ~ArrayDataSource() override;

  bool hasData(const std::string &wsName,
               const std::shared_ptr<Mantid::API::Workspace> &ws) override;

  /// Get DataArray covering full range of data in x, and y directions
  DataArray_const_sptr getDataArray(bool is_log_x) override;

  /// Get DataArray covering restricted range of data
  DataArray_const_sptr getDataArray(double xMin, double xMax, double yMin,
                                    double yMax, size_t nRows, size_t nCols,
                                    bool isLogX) override;

  /// Get a list containing pairs of strings with information about x,y
  std::vector<std::string> getInfoList(double x, double y) override;

private:
  std::vector<float> m_data;
};

using ArrayDataSource_sptr = std::shared_ptr<ArrayDataSource>;
using ArrayDataSource_const_sptr = std::shared_ptr<const ArrayDataSource>;

} // namespace SpectrumView
} // namespace MantidQt
