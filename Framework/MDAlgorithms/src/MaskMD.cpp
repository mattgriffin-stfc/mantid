#include "MantidMDAlgorithms/MaskMD.h"
#include "MantidGeometry/MDGeometry/MDBoxImplicitFunction.h"
#include "MantidAPI/IMDWorkspace.h"
#include "MantidKernel/ArrayProperty.h"
#include "MantidKernel/MandatoryValidator.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>

using namespace Mantid::Kernel;
using namespace Mantid::API;
using namespace Mantid::Geometry;
using boost::regex;

namespace Mantid {
namespace MDAlgorithms {

static const std::string REGEX_STR = "\\[[^\\[\\]]*\\]";

/*
 * Split the input string on commas and trim leading and trailing whitespace
 * from the results
 */
std::vector<std::string> splitByCommas(const std::string &names_string) {
  std::vector<std::string> names_split_by_commas;
  boost::split(names_split_by_commas, names_string, boost::is_any_of(","));

  // Remove leading/trailing whitespace from potential dimension name
  for (auto &name : names_split_by_commas) {
    boost::trim(name);
  }
  return names_split_by_commas;
}

/*
 * Return a vector of only those names from the input string which are within
 * brackets
 */
std::vector<std::string> findNamesInBrackets(const std::string &names_string) {
  std::vector<std::string> names_result;

  regex re(REGEX_STR);
  boost::sregex_token_iterator iter(names_string.begin(), names_string.end(),
                                    re, 0);
  boost::sregex_token_iterator end;
  std::ostringstream ss;
  for (; iter != end; ++iter) {
    ss.str(std::string());
    ss << *iter;
    names_result.push_back(ss.str());
  }

  return names_result;
}

/*
 * Return a vector of names from the string, but with names in brackets reduced
 * to '[]'
 */
std::vector<std::string> removeBracketedNames(const std::string &names_string) {
  regex re(REGEX_STR);
  const std::string names_string_reduced =
      regex_replace(names_string, re, "[]");

  std::vector<std::string> remainder_split =
      splitByCommas(names_string_reduced);

  return remainder_split;
}

/*
 * The list of dimension names often looks like "[H,0,0],[0,K,0]" with "[H,0,0]"
 * being the first dimension but getProperty returns a vector of
 * the string split on every comma
 * This function parses the string, and does not split on commas within brackets
 */
std::vector<std::string> parseDimensionNames(const std::string &names_string) {

  // If there are no brackets then simply split the string on commas
  if (!boost::contains(names_string, "[")) {
    // Split input string by commas
    return splitByCommas(names_string);
  }

  std::vector<std::string> names_result = findNamesInBrackets(names_string);
  std::vector<std::string> remainder_split = removeBracketedNames(names_string);

  // Insert these into results vector if they are not "[]"
  // if they are "[]" then skip a position in the vector instead
  // This preserves the original order of the names,
  // it is also inefficient, but the name list is expected to be short
  size_t names_position = 0;
  auto begin_it = names_result.begin();
  for (std::string name : remainder_split) {
    if (name != "[]") {
      if (names_position == names_result.size()) {
        names_result.push_back(name);
      } else {
        names_result.insert(begin_it + names_position, name);
      }
    }
    ++names_position;
  }
  return names_result;
}

// Register the algorithm into the AlgorithmFactory
DECLARE_ALGORITHM(MaskMD)

/// Local type to group min, max extents with a dimension index.
struct InputArgument {
  double min, max;
  size_t index;
};

/// Comparator to allow sorting by dimension index.
struct LessThanIndex
    : std::binary_function<InputArgument, InputArgument, bool> {
  bool operator()(const InputArgument &a, const InputArgument &b) const {
    return a.index < b.index;
  }
};

//----------------------------------------------------------------------------------------------
/** Constructor
 */
MaskMD::MaskMD() {}

//----------------------------------------------------------------------------------------------
/** Destructor
 */
MaskMD::~MaskMD() {}

//----------------------------------------------------------------------------------------------
/// Algorithm's name for identification. @see Algorithm::name
const std::string MaskMD::name() const { return "MaskMD"; }

/// Algorithm's version for identification. @see Algorithm::version
int MaskMD::version() const { return 1; }

/// Algorithm's category for identification. @see Algorithm::category
const std::string MaskMD::category() const {
  return "MDAlgorithms\\Transforms";
}

//----------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------
/** Initialize the algorithm's properties.
 */
void MaskMD::init() {
  declareProperty(
      new PropertyWithValue<bool>("ClearExistingMasks", true, Direction::Input),
      "Clears any existing masks before applying the provided masking.");
  declareProperty(
      new WorkspaceProperty<IMDWorkspace>("Workspace", "", Direction::InOut),
      "An input/output workspace.");
  declareProperty(
      new ArrayProperty<std::string>(
          "Dimensions",
          boost::make_shared<MandatoryValidator<std::vector<std::string>>>(),
          Direction::Input),
      "Dimension ids/names all comma separated.\n"
      "According to the dimensionality of the workspace, these names will be "
      "grouped,\n"
      "so the number of entries must be n*(number of dimensions in the "
      "workspace).");

  declareProperty(
      new ArrayProperty<double>(
          "Extents",
          boost::make_shared<MandatoryValidator<std::vector<double>>>(),
          Direction::Input),
      "Extents {min, max} corresponding to each of the dimensions specified, "
      "according to the order those identifies have been specified.");
}

/**
Free helper function.
try to fetch the workspace index.
@param ws : The workspace to find the dimensions in
@param candidateNameOrId: Either the name or the id of a dimension in the
workspace.
@return the index of the dimension in the workspace.
@throws runtime_error if the requested dimension is unknown either by id, or by
name in the workspace.
*/
size_t tryFetchDimensionIndex(Mantid::API::IMDWorkspace_sptr ws,
                              const std::string &candidateNameOrId) {
  size_t dimWorkspaceIndex;
  try {
    dimWorkspaceIndex = ws->getDimensionIndexById(candidateNameOrId);
  } catch (std::runtime_error) {
    // this will throw if the name is unknown.
    dimWorkspaceIndex = ws->getDimensionIndexByName(candidateNameOrId);
  }
  return dimWorkspaceIndex;
}

//----------------------------------------------------------------------------------------------
/** Execute the algorithm.
 */
void MaskMD::exec() {
  IMDWorkspace_sptr ws = getProperty("Workspace");
  std::string dimensions_string = getPropertyValue("Dimensions");
  std::vector<double> extents = getProperty("Extents");

  // Dimension names may contain brackets with commas (i.e. [H,0,0])
  // so getProperty would return an incorrect vector of names
  // instead get the string and parse it here
  std::vector<std::string> dimensions = parseDimensionNames(dimensions_string);
  // Report what dimension names were found
  g_log.notice() << "Dimension names parsed as: " << std::endl;
  for (const auto &name : dimensions) {
    g_log.notice() << name << std::endl;
  }

  size_t nDims = ws->getNumDims();
  size_t nDimensionIds = dimensions.size();

  std::stringstream messageStream;

  // Check cardinality on names/ids
  if (nDimensionIds % nDims != 0) {
    messageStream << "Number of dimension ids/names must be n * " << nDims;
    this->g_log.error(messageStream.str());
    throw std::invalid_argument(messageStream.str());
  }

  // Check cardinality on extents
  if (extents.size() != (2 * dimensions.size())) {
    messageStream << "Number of extents must be " << 2 * dimensions.size();
    this->g_log.error(messageStream.str());
    throw std::invalid_argument(messageStream.str());
  }

  // Check extent value provided.
  for (size_t i = 0; i < nDimensionIds; ++i) {
    double min = extents[i * 2];
    double max = extents[(i * 2) + 1];
    if (min > max) {
      messageStream << "Cannot have minimum extents " << min
                    << " larger than maximum extents " << max;
      this->g_log.error(messageStream.str());
      throw std::invalid_argument(messageStream.str());
    }
  }

  size_t nGroups = nDimensionIds / nDims;

  bool bClearExistingMasks = getProperty("ClearExistingMasks");
  if (bClearExistingMasks) {
    ws->clearMDMasking();
  }

  // Loop over all groups
  for (size_t group = 0; group < nGroups; ++group) {
    std::vector<InputArgument> arguments(nDims);

    // Loop over all arguments within the group. and construct InputArguments
    // for sorting.
    for (size_t i = 0; i < nDims; ++i) {
      size_t index = i + (group * nDims);
      InputArgument &arg = arguments[i];

      // Try to get the index of the dimension in the workspace.
      arg.index = tryFetchDimensionIndex(ws, dimensions[index]);

      arg.min = extents[index * 2];
      arg.max = extents[(index * 2) + 1];
    }

    // Sort all the inputs by the dimension index. Without this it will not be
    // possible to construct the MDImplicit function property.
    LessThanIndex comparator;
    std::sort(arguments.begin(), arguments.end(), comparator);

    // Create inputs for a box implicit function
    VMD mins(nDims);
    VMD maxs(nDims);
    for (size_t i = 0; i < nDims; ++i) {
      mins[i] = float(arguments[i].min);
      maxs[i] = float(arguments[i].max);
    }

    // Add new masking.
    ws->setMDMasking(new MDBoxImplicitFunction(mins, maxs));
  }
}

} // namespace Mantid
} // namespace MDAlgorithms
