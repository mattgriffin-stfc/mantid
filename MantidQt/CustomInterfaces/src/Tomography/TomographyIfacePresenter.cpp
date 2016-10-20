#include "MantidAPI/AlgorithmManager.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/TableRow.h"
#include "MantidAPI/WorkspaceGroup.h"
#include "MantidKernel/ConfigService.h"
#include "MantidKernel/FacilityInfo.h"
#include "MantidQtCustomInterfaces/Tomography/ITomographyIfaceView.h"
#include "MantidQtCustomInterfaces/Tomography/TomographyIfaceModel.h"
#include "MantidQtCustomInterfaces/Tomography/TomographyIfacePresenter.h"

#include <boost/lexical_cast.hpp>

#include <Poco/DirectoryIterator.h>
#include <Poco/Path.h>

#include <QFileInfo>
#include <QMutex>
#include <QString>
#include <QThread>
#include <QTimer>

#include "MantidQtCustomInterfaces/Tomography/TomoToolConfigDialogBase.h"

#ifdef _DEBUG
// #define PRESENTER_DEBUG
#endif
using namespace Mantid::API;
using namespace MantidQt::CustomInterfaces;

namespace MantidQt {
namespace CustomInterfaces {

const std::string TomographyIfacePresenter::g_defOutPathLocal =
#ifdef _WIN32
    "D:/imat-data/";
#else
    "~/imat/";
#endif

// TODOPRES: could use ConfigService::isNetworkDrive(const std::string &)
const std::string TomographyIfacePresenter::g_defOutPathRemote =
#ifdef _WIN32
    "I:/imat/imat-data/";
#else
    "~/imat-data/";
#endif

TomographyIfacePresenter::TomographyIfacePresenter(ITomographyIfaceView *view)
    : m_view(view), m_model(new TomographyIfaceModel()), m_statusMutex(NULL),
      m_keepAliveTimer(NULL), m_keepAliveThread(NULL) {
  if (!m_view) {
    throw std::runtime_error("Severe inconsistency found. Presenter created "
                             "with an empty/null view (tomography interface). "
                             "Cannot continue.");
  }
  m_statusMutex = new QMutex();
}

TomographyIfacePresenter::~TomographyIfacePresenter() {
  cleanup();

  if (m_keepAliveThread)
    delete m_keepAliveThread;

  if (m_keepAliveTimer)
    delete m_keepAliveTimer;

  if (m_statusMutex)
    delete m_statusMutex;
}

/**
 * Close open sessions, kill timers/threads etc., save settings, etc. for a
 * graceful window close/destruct
 */
void TomographyIfacePresenter::cleanup() {
  killKeepAliveMechanism();
  m_model->cleanup();
}

void TomographyIfacePresenter::notify(
    ITomographyIfacePresenter::Notification notif) {

  switch (notif) {

  case ITomographyIfacePresenter::SystemSettingsUpdated:
    processSystemSettingsUpdated();
    break;

  case ITomographyIfacePresenter::SetupResourcesAndTools:
    processSetupResourcesAndTools();
    break;

  case ITomographyIfacePresenter::CompResourceChanged:
    processCompResourceChanged();
    break;

  case ITomographyIfacePresenter::ToolChanged:
    processToolChanged();
    break;

  case ITomographyIfacePresenter::TomoPathsChanged:
    processTomoPathsChanged();
    break;

  case ITomographyIfacePresenter::TomoPathsEditedByUser:
    processTomoPathsEditedByUser();
    break;

  case ITomographyIfacePresenter::LogInRequested:
    processLogin();
    break;

  case ITomographyIfacePresenter::LogOutRequested:
    processLogout();
    break;

  case ITomographyIfacePresenter::SetupReconTool:
    processSetupReconTool();
    break;

  case ITomographyIfacePresenter::RunReconstruct:
    processRunRecon();
    break;

  case ITomographyIfacePresenter::RefreshJobs:
    processRefreshJobs();
    break;

  case ITomographyIfacePresenter::CancelJobFromTable:
    processCancelJobs();
    break;

  case ITomographyIfacePresenter::VisualizeJobFromTable:
    processVisualizeJobs();
    break;

  case ITomographyIfacePresenter::ViewImg:
    processViewImg();
    break;

  case ITomographyIfacePresenter::AggregateEnergyBands:
    processAggregateEnergyBands();
    break;

  case ITomographyIfacePresenter::LogMsg:
    processLogMsg();
    break;

  case ITomographyIfacePresenter::ShutDown:
    processShutDown();
    break;
  }
}

void TomographyIfacePresenter::processSystemSettingsUpdated() {
  m_model->setSystemSettings(m_view->systemSettings());

  // update the tool information with the new system settings
  setupConfigDialogSettingsAndUpdateModel(m_configDialog.get());
}

void TomographyIfacePresenter::processSetupResourcesAndTools() {
  // first time update, even if not requested by user
  processSystemSettingsUpdated();

  try {
    m_model->setupComputeResource();
    if (m_model->computeResources().size() < 1) {
      m_view->userWarning("No remote compute resource could be set up!",
                          "No remote compute resource has been set up. Please "
                          "note that without a "
                          "remote compute resource the functionality of this "
                          "interface might be limited.");
    }
    m_model->setupRunTool("");
    processTomoPathsChanged();

    m_view->enableLoggedActions(!m_model->loggedIn().empty());

    m_view->setComputeResources(m_model->computeResources(),
                                m_model->computeResourcesStatus());
    m_view->setReconstructionTools(m_model->reconTools(),
                                   m_model->reconToolsStatus());
  } catch (std::runtime_error &e) {
    const std::string msg =
        "Failed to initialize remote compute resource(s). This "
        "custom interface will not work. Error description: " +
        std::string(e.what());

    m_view->enableLoggedActions(false);
    // This would ideally be shown to the user as a "fatal error" pop-up, as
    // it is an unrecoverable error. But in facilities other than ISIS this
    // would block the builds (docs-qthelp).
    // m_view->userError("Fatal error", msg);
    m_model->logMsg(msg);
  }
}

void TomographyIfacePresenter::processCompResourceChanged() {
  const std::string comp = m_view->currentComputeResource();

  m_model->setupRunTool(comp);

  if ("Local" == comp) {
    m_view->enableLoggedActions(true);
  } else {
    const bool status = !m_model->loggedIn().empty();
    m_view->enableLoggedActions(status);
  }

  // this will udpate the tool after a resource change
  setupConfigDialogSettingsAndUpdateModel(m_configDialog.get());
}

void TomographyIfacePresenter::processToolChanged() {
  const std::string tool = m_view->currentReconTool();

  // disallow reconstruct on tools that don't run yet: Savu and CCPi
  if (TomographyIfaceModel::g_CCPiTool == tool) {
    m_view->enableRunReconstruct(false);
    m_view->enableConfigTool(false);
  } else if (TomographyIfaceModel::g_SavuTool == tool) {
    // for now, show setup dialog, but cannot run
    m_view->enableRunReconstruct(false);
    m_view->enableConfigTool(true);
  } else {
    // No need to be logged in anymore when local is selected
    const bool runningLocally = "Local" == m_view->currentComputeResource();
    m_view->enableRunReconstruct(runningLocally ||
                                 !(m_model->loggedIn().empty()));
    m_view->enableConfigTool(true);
  }

  // return if empty string
  if ("" == tool) {
    return;
  }

  createConfigDialogUsingToolName(tool);
  // this will set the default settings for the dialog
  setupConfigDialogSettingsAndUpdateModel(m_configDialog.get());
}

void TomographyIfacePresenter::setupConfigDialogSettingsAndUpdateModel(
    TomoToolConfigDialogBase *dialog) {

  if (dialog) {
    setupConfigDialogSettings(dialog);
    updateModelAfterToolChanged(dialog);
  }
}

void TomographyIfacePresenter::createConfigDialogUsingToolName(
    const std::string &tool) {
  // free the previous dialogue pointer if any
  m_configDialog.reset();
  m_configDialog = std::unique_ptr<TomoToolConfigDialogBase>(
      TomoToolConfigDialogBase::getCorrectDialogForToolFromString(tool));
}

/**
* Setup the tool's run path, the paths out, the reconstruction index and
* the localOutNameAppendix
*
* @param dialog The dialog pointer that will be set up
*/
void TomographyIfacePresenter::setupConfigDialogSettings(
    TomoToolConfigDialogBase *dialog) {

  if (isLocalResourceSelected()) {
    setupConfigDialogSettingsForLocal(dialog);
  } else {
    setupConfigDialogSettingsForRemote(dialog);
  }
}

void TomographyIfacePresenter::setupConfigDialogSettingsForLocal(
    TomoToolConfigDialogBase *dialog) {
  // set up all the information we need for the dialog
  /* TODOPRES this appends /scripts/ twice, once from C:/MantidInstall/scripts
   and once from /scripts/Imaging/IMAT/tomo_reconstruct.py */
  std::string run = m_model->getExeternalInterpreterPath() + " " +
                    m_model->getCurrentLocalScriptsBasePath() +
                    m_model->getTomoScriptLocationPath();

#ifdef PRESENTER_DEBUG
  std::cout << "DEBUG: LOCAL config dialog settings " << run << "\n";
#endif // PRESENTER_DEBUG

  // TODOPRES get the proper paths for the local reconstruct
  TomoPathsConfig paths = m_view->currentPathsConfig();

  std::string pathOut = Poco::Path::expand(
      g_defOutPathLocal + "/" + m_model->getCurrentExperimentReference());
  static size_t reconIdx = 1;
  const std::string localOutNameAppendix =
      std::string("/processed/") + "reconstruction_" + std::to_string(reconIdx);

  dialog->setupDialog(run, paths, pathOut, localOutNameAppendix);
}

void TomographyIfacePresenter::setupConfigDialogSettingsForRemote(
    TomoToolConfigDialogBase *dialog) {
  // set up all the information we need for the dialog
  std::string run = m_model->getCurrentRemoteScriptsBasePath() + m_model->getTomoScriptFolderPath() +
                    m_model->getTomoScriptLocationPath();
#ifdef PRESENTER_DEBUG
  std::cout << "DEBUG: REMOTE config dialog settings" << run << "\n";
#endif // PRESENTER_DEBUG
  TomoPathsConfig paths = m_view->currentPathsConfig();
  std::string pathOut = Poco::Path::expand(
      g_defOutPathLocal + "/" + m_model->getCurrentExperimentReference());
  static size_t reconIdx = 1;
  const std::string localOutNameAppendix =
      std::string("/processed/") + "reconstruction_" + std::to_string(reconIdx);

  dialog->setupDialog(run, paths, pathOut, localOutNameAppendix);
}

/** Updated all of the model's information about the current tool
*/
void TomographyIfacePresenter::updateModelAfterToolChanged(
    TomoToolConfigDialogBase *dialog) {

  // if passed an empty string don't remove the name
  updateModelCurrentToolName(dialog);
  updateModelCurrentToolMethod(dialog);
  updateModelCurrentToolSettings(dialog);
}

/** Sets the model's current tool name by getting the tool name from the
* dialogue itself
*/
void TomographyIfacePresenter::updateModelCurrentToolName(
    TomoToolConfigDialogBase *dialog) {
  m_model->usingTool(dialog->getSelectedToolName());
}
/** Sets the model's current tool method by coyping the
* string over to the model, getting it from the dialogue itself
*/
void TomographyIfacePresenter::updateModelCurrentToolMethod(
    TomoToolConfigDialogBase *dialog) {
  m_model->setCurrentToolMethod(dialog->getSelectedToolMethod());
}

/** Shares the pointer with the model's tool settings. We don't want a
* unique_ptr, because the dialogs don't die after they are closed, they die if
* the tool is changed or the whole interface is closed
*/
void TomographyIfacePresenter::updateModelCurrentToolSettings(
    TomoToolConfigDialogBase *dialog) {
  m_model->setCurrentToolSettings(dialog->getSelectedToolSettings());
}

/**
 * Simply take the paths that the user has provided.
 */
void TomographyIfacePresenter::processTomoPathsChanged() {
  m_model->setTomoPathsConfig(m_view->currentPathsConfig());
}

/**
* Updates the model with the new path. In the process it also tries to
* guess and find a new path for the flats and darks images from the
* path to the sample images that the user has given. It would normally
* look one level up in the directory tree to see if it can find
* 'dark*' and 'flat*'.
*/
void TomographyIfacePresenter::processTomoPathsEditedByUser() {
  TomoPathsConfig cfg = m_view->currentPathsConfig();
  const std::string samples = cfg.pathSamples();
  if (samples.empty())
    return;

  try {
    findFlatsDarksFromSampleGivenByUser(cfg);
  } catch (std::exception &exc) {
    const std::string msg = "There was a problem while trying to guess the "
                            "location of dark and flat images from this path "
                            "to sample images: " +
                            samples + ". Error details: " + exc.what();
    m_model->logMsg(msg);
  }

  m_model->setTomoPathsConfig(cfg);
  m_view->updatePathsConfig(cfg);
}

void TomographyIfacePresenter::findFlatsDarksFromSampleGivenByUser(
    TomoPathsConfig &cfg) {
  Poco::Path samplesPath(cfg.pathSamples());
  Poco::DirectoryIterator end;
  bool foundDark = false;
  bool foundFlat = false;
  for (Poco::DirectoryIterator it(samplesPath.parent()); it != end; ++it) {
    if (!it->isDirectory()) {
      continue;
    }

    const std::string name = it.name();
    const std::string flatsPrefix = "flat";
    const std::string darksPrefix = "dark";

    if (boost::iequals(name.substr(0, flatsPrefix.length()), flatsPrefix)) {
      cfg.updatePathOpenBeam(it->path(), cfg.m_pathOpenBeamEnabled);
      foundFlat = true;
    } else if (boost::iequals(name.substr(0, darksPrefix.length()),
                              darksPrefix)) {
      cfg.updatePathDarks(it->path(), cfg.m_pathDarkEnabled);
      foundDark = true;
    }
    if (foundFlat && foundDark)
      break;
  }
}

void TomographyIfacePresenter::processLogin() {
  if (!m_model->facilitySupported()) {
    m_view->userError(
        "Fatal error",
        "Cannot do any login operation because the current facility is not "
        "supported by this interface. Please check the log messages for "
        "more "
        "details.");
    return;
  }

  if (!m_model->loggedIn().empty()) {
    m_view->userError(
        "Better to logout before logging in again",
        "You're currently logged in. Please, log out before logging in "
        "again if that's what you meant.");
    return;
  }

  std::string compRes = m_view->currentComputeResource();
  // if local is selected, just take the first remote resource
  if ("Local" == compRes) {
    compRes = "SCARF@STFC";
  }
  try {
    const std::string user = m_view->getUsername();
    if (user.empty()) {
      m_view->userError(
          "Cannot log in",
          "To log in you need to specify a username (and a password).");
      return;
    }

    const std::string passw = m_view->getPassword();
    if (passw.empty()) {
      m_view->userError(
          "Cannot log in with an empty password",
          "Empty passwords are not allowed. Please provide a password).");
      return;
    }

    m_model->doLogin(compRes, user, passw);
  } catch (std::exception &e) {
    throw(std::string("Problem when logging in. Error description: ") +
          e.what());
  }

  bool loggedOK = !m_model->loggedIn().empty();
  m_view->updateLoginControls(loggedOK);
  m_view->enableLoggedActions(loggedOK);

  if (!loggedOK)
    return;

  try {
    m_model->doRefreshJobsInfo(compRes);
  } catch (std::exception &e) {
    throw(std::string("The login operation went apparently fine but an issue "
                      "was found while trying to retrieve the status of the "
                      "jobs currently running on the remote resource. Error "
                      "description: ") +
          e.what());
  }

  startKeepAliveMechanism(m_view->keepAlivePeriod());
  // show table for the first time
  processRefreshJobs();
}

void TomographyIfacePresenter::processLogout() {
  if (m_model->loggedIn().empty()) {
    m_model->logMsg("Cannot log out: not logged into any resource.");
    return;
  }

  try {
    std::string compRes = m_view->currentComputeResource();
    // if local is selected, just take the first remote resource
    if ("Local" == compRes) {
      compRes = "SCARF@STFC";
    }
    m_model->doLogout(compRes, m_view->getUsername());
  } catch (std::exception &e) {
    throw(std::string("Problem when logging out. Error description: ") +
          e.what());
  }

  m_view->updateLoginControls(false);
}

void TomographyIfacePresenter::processSetupReconTool() {
  const std::string &currentReconTool = m_view->currentReconTool();
  // save currentReconTool string to model
  // retrieve, if the same, don't create a new Dialog pointer
  // if not the same, create new Dialog
  if (TomographyIfaceModel::g_CCPiTool != currentReconTool) {

    // TODOPRES save previous state?
    // wrap in unique_ptr

    // give pointer to showToolConfig, so it can run the dialogue
    m_view->showToolConfig(m_configDialog.get());

    updateModelAfterToolChanged(m_configDialog.get());

    // TODOPRES: this would make sense if the reconstruct action/button
    // was only enabled after setting up at least one tool
    // m_view->enableToolsSetupsActions(true);
  }
}

void TomographyIfacePresenter::processRunRecon() {
  // m_model->checkDataPathsSet();
  // TODOPRES: validate data path with additional rules/constraints?
  TomoPathsConfig paths = m_view->currentPathsConfig();
  if (paths.pathSamples().empty()) {
    m_view->userWarning("Sample images path not set!",
                        "The path to the sample images "
                        "is strictly required to start "
                        "a reconstruction");
    return;
  }

  // pre-/post processing steps and filters
  m_model->setPrePostProcSettings(m_view->prePostProcSettings());
  // center of rotation and regions
  m_model->setImageStackPreParams(m_view->currentROIEtcParams());

  try {
    m_model->doSubmitReconstructionJob(m_view->currentComputeResource());
  } catch (std::exception &e) {
    m_view->userWarning("Issue when trying to start a job", e.what());
  }
  processRefreshJobs();
}

bool TomographyIfacePresenter::isLocalResourceSelected() {
  return m_model->localComputeResource() == m_view->currentComputeResource();
}

void TomographyIfacePresenter::processRefreshJobs() {
  // No need to be logged in, there can be local processes
  if (m_model->loggedIn().empty()) {
    m_model->doRefreshJobsInfo("Local");
    m_view->updateJobsInfoDisplay(m_model->jobsStatus(),
                                  m_model->jobsStatusLocal());
    return;
  }

  // TODOPRES doesn't support multiple compute resources
  std::string comp = m_view->currentComputeResource();
  if ("Local" == comp) {
    comp = "SCARF@STFC";
  }
  m_model->doRefreshJobsInfo(comp);

  {
    // update widgets from that info
    QMutexLocker lockit(m_statusMutex);

    m_view->updateJobsInfoDisplay(m_model->jobsStatus(),
                                  m_model->jobsStatusLocal());
  }
}

void TomographyIfacePresenter::processCancelJobs() {
  if (m_model->loggedIn().empty())
    return;

  const std::string &resource = m_view->currentComputeResource();
  if (m_model->localComputeResource() != resource) {
    m_model->doCancelJobs(resource, m_view->processingJobsIDs());
  }
}

void TomographyIfacePresenter::processVisualizeJobs() {
  doVisualize(m_view->processingJobsIDs());
}

void TomographyIfacePresenter::doVisualize(
    const std::vector<std::string> &ids) {
  m_model->logMsg(" Visualizing results from job: " + ids.front());
  // TODOPRES: open dialog, send to Paraview, etc.
}

void TomographyIfacePresenter::processLogMsg() {
  std::vector<std::string> msgs = m_view->logMsgs();
  for (size_t i = 0; i < msgs.size(); i++) {
    m_model->logMsg(msgs[i]);
  }
}

void TomographyIfacePresenter::processAggregateEnergyBands() {
  auto algParams = m_view->currentAggregateBandsParams();

  // check necessary parameters
  if (algParams.end() == algParams.find("InputPath") ||
      algParams.end() == algParams.find("OutputPath") ||
      algParams["InputPath"].empty() || algParams["OutputPath"].empty()) {
    m_view->userError("Invalid input properties",
                      "You need to provide the input properties InputPath and "
                      "OutputPath. Both are mandatory and should point to "
                      "existing directories");
    return;
  }

  // check the paths are usable
  if (!usableEnergyBandsPaths(algParams))
    return;

  const std::string algName = "ImggAggregateWavelengths";
  auto alg = Mantid::API::AlgorithmManager::Instance().create(algName);
  try {
    alg->initialize();
    for (const auto &param : algParams) {
      alg->setPropertyValue(param.first, param.second);
    }
  } catch (std::runtime_error &rexc) {
    m_view->userError("Problem when initializing algorithm",
                      "Could not initialize the algorithm " + algName +
                          " with the options currently set. Error details: " +
                          rexc.what());
  }

  // pass hot potato to the view which has the algorithm runner
  m_view->runAggregateBands(alg);

  m_model->logMsg(" The energy/wavelength bands are being aggregated in the "
                  "background. You can check the log messages and the "
                  "algorithms window to track its progress. ");
}

/**
 * Checks that the input/output directories exists and are readable
 *
 * @param algParams parameters to be passed to the bands aggregation
 * algorithm
 *
 * @return whether it is safe to use the path properties/options given
 */
bool TomographyIfacePresenter::usableEnergyBandsPaths(
    const std::map<std::string, std::string> &algParams) {
  bool usable = false;
  try {
    const std::string name = algParams.at("InputPath");
    Poco::File inPath(name);
    if (!inPath.canRead() || !inPath.isDirectory()) {
      m_view->userError("Invalid input path",
                        "The input path must be a readable directory: " + name);
      return usable;
    }
  } catch (Poco::FileNotFoundException &rexc) {
    m_view->userError("Invalid input path",
                      "The input path must exist on disk. Details: " +
                          std::string(rexc.what()));
    return usable;
  }
  try {
    const std::string name = algParams.at("OutputPath");
    Poco::File outPath(name);
    if (!outPath.canRead() || !outPath.isDirectory()) {
      m_view->userError("Invalid output path",
                        "The output path must be a readable directory: " +
                            name);
      return usable;
    }
  } catch (Poco::FileNotFoundException &rexc) {
    m_view->userError("Invalid output path",
                      "The output path must exist on disk. Details: " +
                          std::string(rexc.what()));
    return usable;
  }

  usable = true;
  return usable;
}

void TomographyIfacePresenter::processShutDown() {
  m_view->saveSettings();
  cleanup();
}

void TomographyIfacePresenter::processViewImg() {
  std::string ip = m_view->showImagePath();
  QString suf = QFileInfo(QString::fromStdString(ip)).suffix();
  // This is not so great, as we check extensions and not really file
  // content/headers, as it should be.
  if ((0 == QString::compare(suf, "fit", Qt::CaseInsensitive)) ||
      (0 == QString::compare(suf, "fits", Qt::CaseInsensitive))) {
    WorkspaceGroup_sptr wsg = m_model->loadFITSImage(ip);
    if (!wsg)
      return;
    MatrixWorkspace_sptr ws =
        boost::dynamic_pointer_cast<MatrixWorkspace>(wsg->getItem(0));
    if (!ws)
      return;

    m_view->showImage(ws);

    // clean-up container group workspace
    if (wsg)
      AnalysisDataService::Instance().remove(wsg->getName());
  } else if ((0 == QString::compare(suf, "tif", Qt::CaseInsensitive)) ||
             (0 == QString::compare(suf, "tiff", Qt::CaseInsensitive)) ||
             (0 == QString::compare(suf, "png", Qt::CaseInsensitive))) {
    m_view->showImage(ip);
  } else {
    m_view->userWarning(
        "Failed to load image - format issue",
        "Could not load image because the extension of the file " + ip +
            ", suffix: " + suf.toStdString() +
            " does not correspond to FITS or TIFF files.");
  }
}

void TomographyIfacePresenter::startKeepAliveMechanism(int period) {
  if (period <= 0) {
    m_model->logMsg("Tomography GUI: not starting the keep-alive mechanism. "
                    "You might be logged out by the remote compute resource "
                    "after some minutes "
                    "depending on system configuration.");
    return;
  }

  m_model->logMsg(
      "Tomography GUI: starting mechanism to periodically query the "
      "status of jobs. This will update the status of running jobs every " +
      boost::lexical_cast<std::string>(period) +
      " seconds. You can also update it at any moment by clicking "
      "on the refresh button. This periodic update mechanism is "
      "also expected to keep sessions on remote compute resources "
      "alive after logging in.");

  if (m_keepAliveThread)
    delete m_keepAliveThread;
  QThread *m_keepAliveThread = new QThread();

  if (m_keepAliveTimer)
    delete m_keepAliveTimer;
  m_keepAliveTimer = new QTimer(NULL); // no-parent so it can be moveToThread

  m_keepAliveTimer->setInterval(1000 * period); // interval in ms
  m_keepAliveTimer->moveToThread(m_keepAliveThread);
  // direct connection from the thread
  connect(m_keepAliveTimer, SIGNAL(timeout()), SLOT(processRefreshJobs()),
          Qt::DirectConnection);
  QObject::connect(m_keepAliveThread, SIGNAL(started()), m_keepAliveTimer,
                   SLOT(start()));
  m_keepAliveThread->start();
}

void TomographyIfacePresenter::killKeepAliveMechanism() {
  if (m_keepAliveTimer)
    m_keepAliveTimer->stop();
}

} // namespace CustomInterfaces
} // namespace MantidQt
