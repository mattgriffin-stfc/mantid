// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidDataHandling/LoadSampleShape.h"
#include "MantidDataHandling/LoadAsciiStl.h"
#include "MantidDataHandling/LoadBinaryStl.h"
#include "MantidDataHandling/LoadOff.h"
#include "MantidDataHandling/LoadStlFactory.h"
#include "MantidGeometry/Instrument/Goniometer.h"

#include "MantidAPI/FileProperty.h"
#include "MantidAPI/InstrumentValidator.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/Run.h"
#include "MantidAPI/Sample.h"

#include "MantidKernel/ArrayProperty.h"

#include "MantidGeometry/Instrument.h"

namespace Mantid {
namespace DataHandling {

// Register the algorithm into the algorithm factory
DECLARE_ALGORITHM(LoadSampleShape)

using namespace Kernel;
using namespace API;
using namespace Geometry;

void LoadSampleShape::init() {
  auto wsValidator = std::make_shared<API::InstrumentValidator>();
  ;

  // input workspace
  declareProperty(
      std::make_unique<WorkspaceProperty<>>("InputWorkspace", "",
                                            Direction::Input, wsValidator),
      "The name of the workspace containing the instrument to add the shape");

  // shape file
  const std::vector<std::string> extensions{".stl", ".off"};
  declareProperty(std::make_unique<FileProperty>(
                      "Filename", "", FileProperty::Load, extensions),
                  "The path name of the file containing the shape");

  // scale to use for stl
  declareProperty("Scale", "cm", "The scale of the stl: m, cm, or mm");

  // Output workspace
  declareProperty(std::make_unique<WorkspaceProperty<>>("OutputWorkspace", "",
                                                        Direction::Output),
                  "The name of the workspace that will contain the loaded "
                  "shape of the sample");
}

void LoadSampleShape::exec() {

  MatrixWorkspace_const_sptr inputWS = getProperty("InputWorkspace");
  MatrixWorkspace_sptr outputWS = getProperty("OutputWorkspace");

  if (inputWS != outputWS) {
    outputWS = inputWS->clone();
  }

  const std::string filename = getProperty("Filename");

  const std::string filetype = filename.substr(filename.size() - 3);

  std::shared_ptr<MeshObject> shape = nullptr;
  const std::string scaleProperty = getPropertyValue("Scale");
  const ScaleUnits scaleType = getScaleTypeFromStr(scaleProperty);

  if (filetype == "off") {
    auto offReader = LoadOff(filename, scaleType);
    shape = offReader.readOFFshape();
  } else {
    std::unique_ptr<LoadStl> reader =
        LoadStlFactory::createReader(filename, scaleType);
    shape = reader->readStl();
  }
  // rotate shape
  rotate(*shape, inputWS);

  // Put shape into sample.
  Sample &sample = outputWS->mutableSample();
  sample.setShape(shape);

  // Set output workspace
  setProperty("OutputWorkspace", outputWS);
}

/**
 * Rotates the Shape by a provided matrix
 * @param sampleMesh The Shape to rotate
 * @param inputWS The workspace to get the rotation from
 * @returns a shared pointer to the newly rotated Shape
 */
void rotate(MeshObject &sampleMesh, const MatrixWorkspace_const_sptr &inputWS) {
  const std::vector<double> rotationMatrix =
      inputWS->run().getGoniometer().getR();
  sampleMesh.rotate(rotationMatrix);
}

} // namespace DataHandling
} // namespace Mantid
