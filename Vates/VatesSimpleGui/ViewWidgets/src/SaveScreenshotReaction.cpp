#include "MantidVatesSimpleGuiViewWidgets/SaveScreenshotReaction.h"

#include "pqActiveObjects.h"
#include "pqApplicationCore.h"
#include "pqCoreUtilities.h"
#include "pqFileDialog.h"
#include "pqProxyWidgetDialog.h"
#include "pqServer.h"
#include "pqSettings.h"
#include "pqView.h"
#include "vtkImageData.h"
#include "vtkNew.h"
#include "vtkSMParaViewPipelineController.h"
#include "vtkSMProperty.h"
#include "vtkSMPropertyHelper.h"
#include "vtkSMSaveScreenshotProxy.h"
#include "vtkSMSessionProxyManager.h"
#include "vtkSMUtilities.h"
#include "vtkSMViewLayoutProxy.h"
#include "vtkSMViewProxy.h"
#include "vtkSmartPointer.h"

#include <QDebug>
#include <QFileInfo>

namespace Mantid {
namespace Vates {
namespace SimpleGui {

//-----------------------------------------------------------------------------
SaveScreenshotReaction::SaveScreenshotReaction(QAction *parentObject)
    : Superclass(parentObject) {
  // load state enable state depends on whether we are connected to an active
  // server or not and whether
  pqActiveObjects *activeObjects = &pqActiveObjects::instance();
  QObject::connect(activeObjects, SIGNAL(serverChanged(pqServer *)), this,
                   SLOT(updateEnableState()));
  QObject::connect(activeObjects, SIGNAL(viewChanged(pqView *)), this,
                   SLOT(updateEnableState()));
  this->updateEnableState();
}

//-----------------------------------------------------------------------------
void SaveScreenshotReaction::updateEnableState() {
  pqActiveObjects *activeObjects = &pqActiveObjects::instance();
  bool is_enabled =
      (activeObjects->activeView() && activeObjects->activeServer());
  this->parentAction()->setEnabled(is_enabled);
}

//-----------------------------------------------------------------------------
QString SaveScreenshotReaction::promptFileName() {
  QString lastUsedExt;
  // Load the most recently used file extensions from QSettings, if available.
  pqSettings *settings = pqApplicationCore::instance()->settings();
  if (settings->contains("extensions/ScreenshotExtension")) {
    lastUsedExt = settings->value("extensions/ScreenshotExtension").toString();
  }

  QString filters("PNG image (*.png);;JPG image (*.jpg);;TIFF image (*.tif)"
                  ";;BMP image (*.bmp);;PPM image (*.ppm)");

  pqFileDialog file_dialog(NULL, pqCoreUtilities::mainWidget(),
                           tr("Save Screenshot:"), QString(), filters);
  file_dialog.setRecentlyUsedExtension(lastUsedExt);
  file_dialog.setObjectName("FileSaveScreenshotDialog");
  file_dialog.setFileMode(pqFileDialog::AnyFile);
  if (file_dialog.exec() != QDialog::Accepted) {
    return QString();
  }

  QString file = file_dialog.getSelectedFiles()[0];
  QFileInfo fileInfo(file);
  lastUsedExt = QString("*.") + fileInfo.suffix();
  settings->setValue("extensions/ScreenshotExtension", lastUsedExt);
  return file;
}

//-----------------------------------------------------------------------------
void SaveScreenshotReaction::saveScreenshot() {
  pqView *view = pqActiveObjects::instance().activeView();
  if (!view) {
    qDebug() << "Cannot save image. No active view.";
    return;
  }

  vtkSMViewProxy *viewProxy = view->getViewProxy();
  vtkSMViewLayoutProxy *layout = vtkSMViewLayoutProxy::FindLayout(viewProxy);
  int showWindowDecorations = -1;

  vtkSMSessionProxyManager *pxm = view->getServer()->proxyManager();
  vtkSmartPointer<vtkSMProxy> proxy;
  proxy.TakeReference(pxm->NewProxy("misc", "SaveScreenshot"));
  vtkSMSaveScreenshotProxy *shProxy =
      vtkSMSaveScreenshotProxy::SafeDownCast(proxy);
  if (!shProxy) {
    qCritical() << "Incorrect type for `SaveScreenshot` proxy.";
    return;
  }

  vtkNew<vtkSMParaViewPipelineController> controller;
  controller->PreInitializeProxy(shProxy);
  vtkSMPropertyHelper(shProxy, "View").Set(viewProxy);
  vtkSMPropertyHelper(shProxy, "Layout").Set(layout);
  controller->PostInitializeProxy(shProxy);

  if (shProxy->UpdateSaveAllViewsPanelVisibility()) {
    Q_ASSERT(layout != NULL);
    // let's hide window decorations.
    vtkSMPropertyHelper helper(layout, "ShowWindowDecorations");
    showWindowDecorations = helper.GetAsInt();
    helper.Set(0);
  }

  pqProxyWidgetDialog dialog(shProxy, pqCoreUtilities::mainWidget());
  dialog.setObjectName("SaveScreenshotDialog");
  dialog.setApplyChangesImmediately(true);
  dialog.setWindowTitle("Save Screenshot Options");
  dialog.setEnableSearchBar(true);
  dialog.setSettingsKey("SaveScreenshotDialog");
  if (dialog.exec() == QDialog::Accepted) {
    QString filename = SaveScreenshotReaction::promptFileName();
    if (!filename.isEmpty()) {
      shProxy->WriteImage(filename.toLocal8Bit().data());
    }
  }

  if (layout && showWindowDecorations != -1) {
    vtkSMPropertyHelper(layout, "ShowWindowDecorations")
        .Set(showWindowDecorations);
    layout->UpdateVTKObjects();
  }

  // This should not be needed as image capturing code only affects back buffer,
  // however it is currently needed due to paraview/paraview#17256. Once that's
  // fixed, we should remove this.
  pqApplicationCore::instance()->render();
}

//-----------------------------------------------------------------------------
bool SaveScreenshotReaction::saveScreenshot(const QString &filename,
                                            const QSize &size, int quality,
                                            bool all_views) {
  pqView *view = pqActiveObjects::instance().activeView();
  if (!view) {
    qDebug() << "Cannot save image. No active view.";
    return false;
  }

  vtkSMViewProxy *viewProxy = view->getViewProxy();

  vtkSmartPointer<vtkImageData> image;
  const vtkVector2i isize(size.width(), size.height());
  if (all_views) {
    vtkSMViewLayoutProxy *layout = vtkSMViewLayoutProxy::FindLayout(viewProxy);
    image = vtkSMSaveScreenshotProxy::CaptureImage(layout, isize);
  } else {
    image = vtkSMSaveScreenshotProxy::CaptureImage(viewProxy, isize);
  }

  if (image) {
    return vtkSMUtilities::SaveImage(image, filename.toLocal8Bit().data(),
                                     quality) != 0;
  }
  return false;
}
}
}
}
