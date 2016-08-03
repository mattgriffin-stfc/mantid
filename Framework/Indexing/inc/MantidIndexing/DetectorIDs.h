#ifndef MANTID_INDEXING_DETECTORIDS_H_
#define MANTID_INDEXING_DETECTORIDS_H_

#include "MantidIndexing/DllConfig.h"

#include <vector>

namespace Mantid {
using detid_t = int32_t;
namespace Indexing {

/** DetectorIDs : TODO: DESCRIPTION

  Copyright &copy; 2016 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
  National Laboratory & European Spallation Source

  This file is part of Mantid.

  Mantid is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  Mantid is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.

  File change history is stored at: <https://github.com/mantidproject/mantid>
  Code Documentation is available at: <http://doxygen.mantidproject.org>
*/
class MANTID_INDEXING_DLL DetectorIDs {
public:
  explicit DetectorIDs(const std::vector<detid_t> &detectorIDs);
  explicit DetectorIDs(std::vector<std::vector<detid_t>> detectorIDs);
  explicit DetectorIDs(std::initializer_list<detid_t> &&ilist);

  size_t size() const;
  const std::vector<std::vector<detid_t>> &data() const;

private:
  std::vector<std::vector<detid_t>> m_data;
};

} // namespace Indexing
} // namespace Mantid

#endif /* MANTID_INDEXING_DETECTORIDS_H_ */
