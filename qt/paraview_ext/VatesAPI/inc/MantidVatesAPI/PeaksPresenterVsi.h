// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidAPI/IPeaksWorkspace_fwd.h"
#include "MantidKernel/SpecialCoordinateSystem.h"
#include "MantidKernel/System.h"
#include "MantidKernel/V3D.h"
#include "MantidVatesAPI/ViewFrustum.h"
#include <string>
#include <vector>

namespace Mantid {
namespace VATES {

class DLLExport PeaksPresenterVsi {
public:
  virtual ~PeaksPresenterVsi(){};
  virtual std::vector<bool> getViewablePeaks() const = 0;
  virtual Mantid::API::IPeaksWorkspace_sptr getPeaksWorkspace() const = 0;
  virtual void updateViewFrustum(ViewFrustum_const_sptr frustum) = 0;
  virtual std::string getFrame() const = 0;
  virtual std::string getPeaksWorkspaceName() const = 0;
  virtual void
  getPeaksInfo(Mantid::API::IPeaksWorkspace_sptr peaksWorkspace, int row,
               Mantid::Kernel::V3D &position, double &radius,
               Mantid::Kernel::SpecialCoordinateSystem specialCoordinateSystem)
      const = 0;
  virtual void sortPeaksWorkspace(const std::string &byColumnName,
                                  const bool ascending) = 0;
};

using PeaksPresenterVsi_sptr = std::shared_ptr<PeaksPresenterVsi>;
using PeaksPresenterVsi_const_sptr = std::shared_ptr<const PeaksPresenterVsi>;
} // namespace VATES
} // namespace Mantid