#include "MantidQtWidgets/Common/Batch/FindSubtreeRoots.h"
namespace MantidQt {
namespace MantidWidgets {
namespace Batch {

auto FindSubtreeRoots::operator()(std::vector<RowLocation> region)
    -> boost::optional<std::vector<RowLocation>> {
  std::sort(region.begin(), region.end());
  if (!region.empty()) {
    nodeWasSubtreeRoot(*region.cbegin());
    auto roots = std::vector<RowLocation>({previousNode});
    auto current = region.begin() + 1;
    auto previousRoot = previousNode;

    for (; current != region.end(); ++current) {
      auto &currentNode = *current;
      if (isChildOfPrevious(currentNode) ||
          (!previousWasRoot && isSiblingOfPrevious(currentNode))) {
        nodeWasNotSubtreeRoot(currentNode);
      } else if (currentNode.isDescendantOf(previousRoot)) {
        if (previousNode.depth() < currentNode.depth())
          return boost::none;
        else
          nodeWasNotSubtreeRoot(currentNode);
      } else {
        if (previousRoot.depth() > currentNode.depth())
          return boost::none;
        else {
          nodeWasSubtreeRoot(currentNode);
          roots.emplace_back(currentNode);
          previousRoot = std::move(currentNode);
        }
      }
    }
    return roots;
  } else {
    return std::vector<RowLocation>();
  }
}

void FindSubtreeRoots::nodeWasSubtreeRoot(RowLocation const &rowLocation) {
  previousWasRoot = true;
  previousNode = rowLocation;
}

void FindSubtreeRoots::nodeWasNotSubtreeRoot(RowLocation const &rowLocation) {
  previousWasRoot = false;
  previousNode = rowLocation;
}

bool FindSubtreeRoots::isChildOfPrevious(RowLocation const &location) const {
  return location.isChildOf(previousNode);
}

bool FindSubtreeRoots::isSiblingOfPrevious(RowLocation const &location) const {
  return location.isSiblingOf(previousNode);
}

} // namespace Batch
} // namespace MantidWidgets
} // namespace MantidQt
