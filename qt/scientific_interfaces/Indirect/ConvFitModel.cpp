// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "ConvFitModel.h"

#include "MantidAPI/AlgorithmManager.h"
#include "MantidAPI/FunctionFactory.h"
#include "MantidAPI/MultiDomainFunction.h"
#include "MantidGeometry/Instrument.h"

#include <boost/algorithm/string/join.hpp>

#include <stdexcept>
#include <utility>

using namespace Mantid::API;

namespace {

MatrixWorkspace_sptr getADSMatrixWorkspace(std::string const &workspaceName) {
  return AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>(
      workspaceName);
}

bool doesExistInADS(std::string const &workspaceName) {
  return AnalysisDataService::Instance().doesExist(workspaceName);
}

IAlgorithm_sptr loadParameterFileAlgorithm(std::string const &workspaceName,
                                           std::string const &filename) {
  auto loadParamFile = AlgorithmManager::Instance().create("LoadParameterFile");
  loadParamFile->initialize();
  loadParamFile->setProperty("Workspace", workspaceName);
  loadParamFile->setProperty("Filename", filename);
  return loadParamFile;
}

void readAnalyserFromFile(const std::string &analyser,
                          const MatrixWorkspace_sptr &workspace) {
  auto const instrument = workspace->getInstrument();
  auto const idfDirectory = Mantid::Kernel::ConfigService::Instance().getString(
      "instrumentDefinition.directory");
  auto const reflection = instrument->getStringParameter("reflection")[0];
  auto const parameterFile = idfDirectory + instrument->getName() + "_" +
                             analyser + "_" + reflection + "_Parameters.xml";

  auto loadParamFile =
      loadParameterFileAlgorithm(workspace->getName(), parameterFile);
  loadParamFile->execute();

  if (!loadParamFile->isExecuted())
    throw std::invalid_argument(
        "Could not load parameter file, ensure instrument "
        "directory is in data search paths.");
}

Mantid::Geometry::IComponent_const_sptr
getAnalyser(const MatrixWorkspace_sptr &workspace) {
  auto const instrument = workspace->getInstrument();
  auto const analysers = instrument->getStringParameter("analyser");

  if (analysers.empty())
    throw std::invalid_argument(
        "Could not load instrument resolution from parameter file");

  auto const component = instrument->getComponentByName(analysers[0]);
  if (component) {
    if (component->hasParameter("resolution")) {
      auto const resolutionParameters =
          component->getNumberParameter("resolution");
      if (resolutionParameters.empty())
        readAnalyserFromFile(analysers[0], workspace);
    }
  } else {
    readAnalyserFromFile(analysers[0], workspace);
  }
  return instrument->getComponentByName(analysers[0]);
}

boost::optional<double>
instrumentResolution(const MatrixWorkspace_sptr &workspace) {
  try {
    auto const analyser = getAnalyser(workspace);
    if (analyser && analyser->hasParameter("resolution"))
      return analyser->getNumberParameter("resolution")[0];

    auto const instrument = workspace->getInstrument();
    if (instrument && instrument->hasParameter("resolution"))
      return instrument->getNumberParameter("resolution")[0];
    else if (instrument && instrument->hasParameter("EFixed"))
      return instrument->getNumberParameter("EFixed")[0] * 0.01;

    return boost::none;
  } catch (Mantid::Kernel::Exception::NotFoundError const &) {
    return boost::none;
  } catch (std::invalid_argument const &) {
    return boost::none;
  } catch (std::exception const &) {
    return boost::none;
  }
}

MatrixWorkspace_sptr cloneWorkspace(const MatrixWorkspace_sptr &inputWS,
                                    std::string const &outputName) {
  auto cloneAlg = AlgorithmManager::Instance().create("CloneWorkspace");
  cloneAlg->setLogging(false);
  cloneAlg->initialize();
  cloneAlg->setProperty("InputWorkspace", inputWS);
  cloneAlg->setProperty("OutputWorkspace", outputName);
  cloneAlg->execute();
  return getADSMatrixWorkspace(outputName);
}

MatrixWorkspace_sptr appendWorkspace(const MatrixWorkspace_sptr &leftWS,
                                     const MatrixWorkspace_sptr &rightWS,
                                     int numHistograms,
                                     std::string const &outputName) {
  auto appendAlg = AlgorithmManager::Instance().create("AppendSpectra");
  appendAlg->setLogging(true);
  appendAlg->initialize();
  appendAlg->setProperty("InputWorkspace1", leftWS);
  appendAlg->setProperty("InputWorkspace2", rightWS);
  appendAlg->setProperty("Number", numHistograms);
  appendAlg->setProperty("OutputWorkspace", outputName);
  appendAlg->execute();
  return getADSMatrixWorkspace(outputName);
}

void renameWorkspace(std::string const &name, std::string const &newName) {
  auto renamer = AlgorithmManager::Instance().create("RenameWorkspace");
  renamer->setLogging(false);
  renamer->setProperty("InputWorkspace", name);
  renamer->setProperty("OutputWorkspace", newName);
  renamer->execute();
}

void deleteWorkspace(std::string const &workspaceName) {
  if (!AnalysisDataService::Instance().doesExist(workspaceName))
    return;
  auto deleter = AlgorithmManager::Instance().create("DeleteWorkspace");
  deleter->setLogging(false);
  deleter->setProperty("Workspace", workspaceName);
  deleter->execute();
}

void extendResolutionWorkspace(const MatrixWorkspace_sptr &resolution,
                               std::size_t numberOfHistograms,
                               std::string const &outputName) {
  const auto resolutionNumHist = resolution->getNumberHistograms();
  if (resolutionNumHist != 1 && resolutionNumHist != numberOfHistograms) {
    std::string msg(
        "Resolution must have either one or as many spectra as the sample");
    throw std::runtime_error(msg);
  }

  auto resolutionWS = cloneWorkspace(resolution, "__cloned");

  // Append to cloned workspace if necessary
  if (resolutionNumHist == 1 && numberOfHistograms > 1) {
    appendWorkspace(resolutionWS, resolution,
                    static_cast<int>(numberOfHistograms - 1), outputName);
    deleteWorkspace("__cloned");
  } else
    renameWorkspace("__cloned", outputName);
}

void getParameterNameChanges(
    const IFunction &model, const std::string &oldPrefix,
    const std::string &newPrefix,
    std::unordered_map<std::string, std::string> &changes) {
  for (const auto &parameterName : model.getParameterNames())
    changes[newPrefix + parameterName] = oldPrefix + parameterName;
}

void getParameterNameChanges(
    const CompositeFunction &model, const std::string &prefixPrefix,
    const std::string &prefixSuffix, std::size_t from, std::size_t to,
    std::unordered_map<std::string, std::string> &changes) {
  size_t di = from > 0 ? 1 : 0;
  for (auto i = from; i < to; ++i) {
    const auto oldPrefix = "f" + std::to_string(i) + ".";
    const auto functionPrefix = "f" + std::to_string(i - di) + ".";
    const auto function = model.getFunction(i);
    auto newPrefix = prefixPrefix + functionPrefix;

    if (function->name() != "Delta Function")
      newPrefix += prefixSuffix;

    getParameterNameChanges(*function, oldPrefix, newPrefix, changes);
  }
}

std::unordered_map<std::string, std::string> parameterNameChanges(
    const CompositeFunction &model, const std::string &prefixPrefix,
    const std::string &prefixSuffix, std::size_t backgroundIndex) {
  std::unordered_map<std::string, std::string> changes;
  auto const nFunctions = model.nFunctions();
  if (nFunctions > 2) {
    getParameterNameChanges(model, prefixPrefix, prefixSuffix, 0,
                            backgroundIndex, changes);

    const auto backgroundPrefix = "f" + std::to_string(backgroundIndex) + ".";
    getParameterNameChanges(*model.getFunction(backgroundIndex),
                            backgroundPrefix, "f0.", changes);

    getParameterNameChanges(model, prefixPrefix, prefixSuffix,
                            backgroundIndex + 1, model.nFunctions(), changes);
  } else if (nFunctions == 2) {
    const auto backgroundPrefix = "f" + std::to_string(backgroundIndex) + ".";
    getParameterNameChanges(*model.getFunction(backgroundIndex),
                            backgroundPrefix, "f0.", changes);
    size_t otherIndex = backgroundIndex == 0 ? 1 : 0;
    const auto otherPrefix = "f" + std::to_string(otherIndex) + ".";
    getParameterNameChanges(*model.getFunction(otherIndex), otherPrefix,
                            prefixPrefix, changes);
  } else {
    throw std::runtime_error(
        "Composite function is expected to have more than 1 member.");
  }
  return changes;
}

std::unordered_map<std::string, std::string>
parameterNameChanges(const CompositeFunction &model,
                     const std::string &prefixPrefix,
                     const std::string &prefixSuffix) {
  std::unordered_map<std::string, std::string> changes;
  getParameterNameChanges(model, prefixPrefix, prefixSuffix, 0,
                          model.nFunctions(), changes);
  return changes;
}

std::unordered_map<std::string, std::string>
parameterNameChanges(const IFunction &model, const std::string &prefixPrefix,
                     const std::string &prefixSuffix) {
  std::unordered_map<std::string, std::string> changes;
  getParameterNameChanges(model, "", prefixPrefix + prefixSuffix, changes);
  return changes;
}

std::unordered_map<std::string, std::string>
constructParameterNameChanges(const IFunction &model,
                              boost::optional<std::size_t> backgroundIndex,
                              bool temperatureUsed) {
  std::string prefixPrefix = backgroundIndex ? "f1.f1." : "f1.";
  std::string prefixSuffix = temperatureUsed ? "f1." : "";

  try {
    const auto &compositeModel = dynamic_cast<const CompositeFunction &>(model);
    if (backgroundIndex)
      return parameterNameChanges(compositeModel, prefixPrefix, prefixSuffix,
                                  *backgroundIndex);
    return parameterNameChanges(compositeModel, prefixPrefix, prefixSuffix);
  } catch (const std::bad_cast &) {
    return parameterNameChanges(model, prefixPrefix, prefixSuffix);
  }
}

IAlgorithm_sptr addSampleLogAlgorithm(const Workspace_sptr &workspace,
                                      const std::string &name,
                                      const std::string &text,
                                      const std::string &type) {
  auto addSampleLog = AlgorithmManager::Instance().create("AddSampleLog");
  addSampleLog->setLogging(false);
  addSampleLog->setProperty("Workspace", workspace->getName());
  addSampleLog->setProperty("LogName", name);
  addSampleLog->setProperty("LogText", text);
  addSampleLog->setProperty("LogType", type);
  return addSampleLog;
}

struct AddSampleLogRunner {
  AddSampleLogRunner(Workspace_sptr resultWorkspace,
                     WorkspaceGroup_sptr resultGroup)
      : m_resultWorkspace(std::move(resultWorkspace)),
        m_resultGroup(std::move(resultGroup)) {}

  void operator()(const std::string &name, const std::string &text,
                  const std::string &type) {
    addSampleLogAlgorithm(m_resultWorkspace, name, text, type)->execute();
    addSampleLogAlgorithm(m_resultGroup, name, text, type)->execute();
  }

private:
  Workspace_sptr m_resultWorkspace;
  WorkspaceGroup_sptr m_resultGroup;
};

std::vector<std::string>
getNames(const MantidQt::CustomInterfaces::IDA::ResolutionCollectionType
             &workspaces) {
  std::vector<std::string> names;
  names.reserve(workspaces.size().value);
  std::transform(
      workspaces.begin(), workspaces.end(), std::back_inserter(names),
      [](const std::weak_ptr<Mantid::API::MatrixWorkspace> &workspace) {
        return workspace.lock()->getName();
      });
  return names;
}

void setResolutionAttribute(const CompositeFunction_sptr &convolutionModel,
                            const IFunction::Attribute &attr) {
  if (convolutionModel->name() == "Convolution")
    convolutionModel->getFunction(0)->setAttribute("Workspace", attr);
  else {
    auto convolution = std::dynamic_pointer_cast<CompositeFunction>(
        convolutionModel->getFunction(1));
    convolution->getFunction(0)->setAttribute("Workspace", attr);
  }
}
} // namespace

namespace MantidQt {
namespace CustomInterfaces {
namespace IDA {

ConvFitModel::ConvFitModel() {}

ConvFitModel::~ConvFitModel() {
  for (const auto &resolution : m_extendedResolution)
    AnalysisDataService::Instance().remove(resolution);
}

IAlgorithm_sptr ConvFitModel::sequentialFitAlgorithm() const {
  return AlgorithmManager::Instance().create("ConvolutionFitSequential");
}

IAlgorithm_sptr ConvFitModel::simultaneousFitAlgorithm() const {
  return AlgorithmManager::Instance().create("ConvolutionFitSimultaneous");
}

std::string ConvFitModel::sequentialFitOutputName() const {
  if (isMultiFit())
    return "MultiConvFit_seq_" + m_fitType + m_backgroundString + "_Results";
  return createOutputName("%1%_conv_seq_" + m_fitType + m_backgroundString +
                              "_s%2%",
                          "_to_", TableDatasetIndex{0});
}

std::string ConvFitModel::simultaneousFitOutputName() const {
  if (isMultiFit())
    return "MultiConvFit_sim_" + m_fitType + m_backgroundString + "_Results";
  return createOutputName("%1%_conv_sim_" + m_fitType + m_backgroundString +
                              "_s%2%",
                          "_to_", TableDatasetIndex{0});
}

std::string ConvFitModel::singleFitOutputName(TableDatasetIndex index,
                                              WorkspaceIndex spectrum) const {
  return createSingleFitOutputName("%1%_conv_" + m_fitType +
                                       m_backgroundString + "_s%2%_Results",
                                   index, spectrum);
}

Mantid::API::MultiDomainFunction_sptr ConvFitModel::getFittingFunction() const {
  return IndirectFittingModel::getFittingFunction();
}

boost::optional<double>
ConvFitModel::getInstrumentResolution(TableDatasetIndex dataIndex) const {
  if (dataIndex < numberOfWorkspaces())
    return instrumentResolution(getWorkspace(dataIndex));
  return boost::none;
}

std::size_t ConvFitModel::getNumberHistograms(TableDatasetIndex index) const {
  return getWorkspace(index)->getNumberHistograms();
}

MatrixWorkspace_sptr
ConvFitModel::getResolution(TableDatasetIndex index) const {
  if (index < m_resolution.size())
    return m_resolution[index].lock();
  return nullptr;
}

MultiDomainFunction_sptr ConvFitModel::getMultiDomainFunction() const {
  auto function = IndirectFittingModel::getMultiDomainFunction();
  const std::string base = "__ConvFitResolution";

  for (auto i = 0u; i < function->nFunctions(); ++i)
    setResolutionAttribute(function,
                           IFunction::Attribute(base + std::to_string(i)));
  return function;
}

std::vector<std::string> ConvFitModel::getSpectrumDependentAttributes() const {
  /// Q value also depends on spectrum but is automatically updated when
  /// the WorkspaceIndex is changed
  return {"WorkspaceIndex"};
}

void ConvFitModel::setFitFunction(MultiDomainFunction_sptr function) {
  IndirectFittingModel::setFitFunction(function);
}

void ConvFitModel::setTemperature(const boost::optional<double> &temperature) {
  m_temperature = temperature;
}

void ConvFitModel::addWorkspace(MatrixWorkspace_sptr workspace,
                                const Spectra &spectra) {
  IndirectFittingModel::addWorkspace(workspace, spectra);
}

void ConvFitModel::removeWorkspace(TableDatasetIndex index) {
  IndirectFittingModel::removeWorkspace(index);

  const auto newSize = numberOfWorkspaces();
  while (m_resolution.size() > newSize)
    m_resolution.remove(index);

  while (m_extendedResolution.size() > newSize) {
    AnalysisDataService::Instance().remove(m_extendedResolution[index]);
    m_extendedResolution.remove(index);
  }
}

void ConvFitModel::setResolution(const std::string &name,
                                 TableDatasetIndex index) {
  if (!name.empty() && doesExistInADS(name))
    setResolution(getADSMatrixWorkspace(name), index);
  else
    throw std::runtime_error("A valid resolution file needs to be selected.");
}

void ConvFitModel::setResolution(const MatrixWorkspace_sptr &resolution,
                                 TableDatasetIndex index) {
  if (m_resolution.size() > index)
    m_resolution[index] = resolution;
  else if (m_resolution.size() == index)
    m_resolution.emplace_back(resolution);
  else
    throw std::out_of_range("Provided resolution index '" +
                            std::to_string(index.value) +
                            "' was out of range.");

  if (numberOfWorkspaces() > index)
    addExtendedResolution(index);
}

void ConvFitModel::addExtendedResolution(TableDatasetIndex index) {
  const std::string name = "__ConvFitResolution" + std::to_string(index.value);

  extendResolutionWorkspace(m_resolution[index].lock(),
                            getNumberHistograms(index), name);

  if (m_extendedResolution.size() > index)
    m_extendedResolution[index] = name;
  else
    m_extendedResolution.emplace_back(name);
}

void ConvFitModel::setFitTypeString(const std::string &fitType) {
  m_fitType = fitType;
}

std::unordered_map<std::string, ParameterValue>
ConvFitModel::createDefaultParameters(TableDatasetIndex index) const {
  std::unordered_map<std::string, ParameterValue> defaultValues;
  defaultValues["PeakCentre"] = ParameterValue(0.0);
  defaultValues["Centre"] = ParameterValue(0.0);
  // Reset all parameters to default of 1
  defaultValues["Amplitude"] = ParameterValue(1.0);
  defaultValues["beta"] = ParameterValue(1.0);
  defaultValues["Decay"] = ParameterValue(1.0);
  defaultValues["Diffusion"] = ParameterValue(1.0);
  defaultValues["Height"] = ParameterValue(1.0);
  defaultValues["Intensity"] = ParameterValue(1.0);
  defaultValues["Radius"] = ParameterValue(1.0);
  defaultValues["Tau"] = ParameterValue(1.0);

  auto resolution = getInstrumentResolution(index);
  if (resolution)
    defaultValues["FWHM"] = ParameterValue(*resolution);
  return defaultValues;
}

std::unordered_map<std::string, std::string>
ConvFitModel::mapDefaultParameterNames() const {
  const auto initialMapping = IndirectFittingModel::mapDefaultParameterNames();
  std::unordered_map<std::string, std::string> mapping;
  for (const auto &map : initialMapping) {
    auto mapped = m_parameterNameChanges.find(map.second);
    if (mapped != m_parameterNameChanges.end())
      mapping[map.first] = mapped->second;
    else
      mapping.insert(map);
  }
  return mapping;
}

void ConvFitModel::addSampleLogs() {
  AddSampleLogRunner addSampleLog(getResultWorkspace(), getResultGroup());
  addSampleLog("resolution_filename",
               boost::algorithm::join(getNames(m_resolution), ","), "String");

  if (m_temperature && m_temperature.get() != 0.0) {
    addSampleLog("temperature_correction", "true", "String");
    addSampleLog("temperature_value", std::to_string(m_temperature.get()),
                 "Number");
  }
}

IndirectFitOutput ConvFitModel::createFitOutput(
    WorkspaceGroup_sptr resultGroup, ITableWorkspace_sptr parameterTable,
    WorkspaceGroup_sptr resultWorkspace, const FitDataIterator &fitDataBegin,
    const FitDataIterator &fitDataEnd) const {
  auto output = IndirectFitOutput(resultGroup, parameterTable, resultWorkspace,
                                  fitDataBegin, fitDataEnd);
  output.mapParameterNames(m_parameterNameChanges, fitDataBegin, fitDataEnd);
  return output;
}

IndirectFitOutput
ConvFitModel::createFitOutput(Mantid::API::WorkspaceGroup_sptr resultGroup,
                              Mantid::API::ITableWorkspace_sptr parameterTable,
                              Mantid::API::WorkspaceGroup_sptr resultWorkspace,
                              IndirectFitData *fitData,
                              WorkspaceIndex spectrum) const {
  auto output = IndirectFitOutput(resultGroup, parameterTable, resultWorkspace,
                                  fitData, spectrum);
  output.mapParameterNames(m_parameterNameChanges, fitData, spectrum);
  return output;
}

void ConvFitModel::addOutput(Mantid::API::IAlgorithm_sptr fitAlgorithm) {
  IndirectFittingModel::addOutput(fitAlgorithm);
  addSampleLogs();
}

void ConvFitModel::addOutput(IndirectFitOutput *fitOutput,
                             WorkspaceGroup_sptr resultGroup,
                             ITableWorkspace_sptr parameterTable,
                             WorkspaceGroup_sptr resultWorkspace,
                             const FitDataIterator &fitDataBegin,
                             const FitDataIterator &fitDataEnd) const {
  fitOutput->addOutput(resultGroup, parameterTable, resultWorkspace,
                       fitDataBegin, fitDataEnd);
  fitOutput->mapParameterNames(m_parameterNameChanges, fitDataBegin,
                               fitDataEnd);
}

void ConvFitModel::addOutput(IndirectFitOutput *fitOutput,
                             WorkspaceGroup_sptr resultGroup,
                             ITableWorkspace_sptr parameterTable,
                             WorkspaceGroup_sptr resultWorkspace,
                             IndirectFitData *fitData,
                             WorkspaceIndex spectrum) const {
  fitOutput->addOutput(resultGroup, parameterTable, resultWorkspace, fitData,
                       spectrum);
  fitOutput->mapParameterNames(m_parameterNameChanges, fitData, spectrum);
}

void ConvFitModel::setParameterNameChanges(
    const IFunction &model, boost::optional<std::size_t> backgroundIndex) {
  m_parameterNameChanges = constructParameterNameChanges(
      model, std::move(backgroundIndex), m_temperature.is_initialized());
}

std::vector<std::pair<std::string, int>>
ConvFitModel::getResolutionsForFit() const {
  std::vector<std::pair<std::string, int>> resolutionVector;
  for (TableDatasetIndex index = TableDatasetIndex{0};
       index < m_resolution.size(); ++index) {

    auto spectra = getSpectra(index);
    auto singleSpectraResolution =
        getResolution(index)->getNumberHistograms() == 1;
    for (auto &spectraIndex : spectra) {
      auto resolutionIndex = singleSpectraResolution ? 0 : spectraIndex.value;
      resolutionVector.emplace_back(
          std::make_pair(getResolution(index)->getName(), resolutionIndex));
    }
  }
  return resolutionVector;
}

} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt
