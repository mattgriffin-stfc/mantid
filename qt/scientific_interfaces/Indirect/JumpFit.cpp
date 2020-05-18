// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "JumpFit.h"
#include "FQFitConstants.h"
#include "IndirectFunctionBrowser/SingleFunctionTemplateBrowser.h"
#include "JumpFitDataPresenter.h"

#include "MantidAPI/AlgorithmManager.h"
#include "MantidAPI/FunctionFactory.h"
#include "MantidAPI/IFunction.h"
#include "MantidAPI/ITableWorkspace.h"
#include "MantidAPI/MultiDomainFunction.h"
#include "MantidAPI/Run.h"
#include "MantidAPI/WorkspaceGroup.h"
#include "MantidQtWidgets/Common/UserInputValidator.h"

#include "MantidQtWidgets/Common/SignalBlocker.h"

#include <boost/algorithm/string/join.hpp>

#include <string>

using namespace Mantid::API;

namespace MantidQt {
namespace CustomInterfaces {
namespace IDA {

JumpFit::JumpFit(QWidget *parent)
    : IndirectFitAnalysisTab(new JumpFitModel, parent),
      m_uiForm(new Ui::JumpFit) {
  m_uiForm->setupUi(parent);

  m_jumpFittingModel = dynamic_cast<JumpFitModel *>(fittingModel());
  auto templateBrowser = new SingleFunctionTemplateBrowser(widthFits);
  setPlotView(m_uiForm->pvFitPlotView);
  setFitDataPresenter(std::make_unique<JumpFitDataPresenter>(
      m_jumpFittingModel, m_uiForm->fitDataView, m_uiForm->cbParameterType,
      m_uiForm->cbParameter, m_uiForm->lbParameterType, m_uiForm->lbParameter,
      templateBrowser));

  setSpectrumSelectionView(m_uiForm->svSpectrumView);
  setOutputOptionsView(m_uiForm->ovOutputOptionsView);

  m_uiForm->fitPropertyBrowser->setFunctionTemplateBrowser(templateBrowser);
  setFitPropertyBrowser(m_uiForm->fitPropertyBrowser);

  setEditResultVisible(false);
  m_uiForm->fitDataView->setStartAndEndHidden(false);
}

void JumpFit::setupFitTab() {
  m_uiForm->svSpectrumView->hideSpectrumSelector();
  m_uiForm->svSpectrumView->hideMaskSpectrumSelector();

  m_uiForm->cbParameter->setEnabled(false);

  connect(m_uiForm->pbRun, SIGNAL(clicked()), this, SLOT(runClicked()));
  connect(this, SIGNAL(functionChanged()), this,
          SLOT(updateModelFitTypeString()));
}

void JumpFit::updateModelFitTypeString() {
  m_jumpFittingModel->setFitType(fitTypeString());
}

std::string JumpFit::fitTypeString() const {
  if (!m_jumpFittingModel->getFittingFunction() ||
      m_jumpFittingModel->getFittingFunction()->nFunctions() == 0) {
    return "NoCurrentFunction";
  }

  auto fun = m_jumpFittingModel->getFittingFunction()->getFunction(0);
  if (fun->nFunctions() == 0) {
    return fun->name();
  } else {
    return "UserDefinedCompositeFunction";
  }
}

void JumpFit::runClicked() { runTab(); }

void JumpFit::setRunIsRunning(bool running) {
  m_uiForm->pbRun->setText(running ? "Running..." : "Run");
}

void JumpFit::setRunEnabled(bool enable) {
  m_uiForm->pbRun->setEnabled(enable);
}

EstimationDataSelector JumpFit::getEstimationDataSelector() const {
  return [](const std::vector<double> &,
            const std::vector<double> &) -> DataForParameterEstimation {
    return DataForParameterEstimation{};
  };
}

} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt
