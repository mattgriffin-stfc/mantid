//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
#include "MantidAlgorithms/ConvertUnits.h"
#include "MantidAPI/AlgorithmFactory.h"
#include "MantidKernel/UnitFactory.h"

namespace Mantid
{
namespace Algorithms
{

// Register with the algorithm factory
	DECLARE_NAMESPACED_ALGORITHM(Mantid::Algorithms,ConvertUnits)

using namespace Kernel;
using namespace API;

// Get a reference to the logger
Logger& ConvertUnits::g_log = Logger::get("ConvertUnits");

/// Default constructor
ConvertUnits::ConvertUnits() : Algorithm()
{
}

/// Destructor
ConvertUnits::~ConvertUnits()
{
}

/// Initialisation method
void ConvertUnits::init()
{
  declareProperty(new WorkspaceProperty<API::Workspace>("InputWorkspace","",Direction::Input));
  declareProperty(new WorkspaceProperty<API::Workspace>("OutputWorkspace","",Direction::Output));  
  declareProperty("Target","",new MandatoryValidator);
  declareProperty("Emode",0);
  declareProperty("Efixed",0.0);
}

/** Executes the algorithm
 *  @throw std::runtime_error If the input workspace has not had its unit set
 *  @throw NotImplementedError If the input workspace contains point (not histogram) data
 *  @throw InstrumentDefinitionError If unable to calculate source-sample distance
 */
void ConvertUnits::exec()
{
  // Get the workspace
  API::Workspace_sptr inputWS = getProperty("InputWorkspace");
  
  // Check that the workspace is histogram data
  if (inputWS->dataX(0).size() == inputWS->dataY(0).size())
  {
    g_log.error("Conversion of units for point data is not yet implemented");
    throw Exception::NotImplementedError("Conversion of units for point data is not yet implemented");
  }
  
  // Check that the input workspace has had its unit set
  if ( ! inputWS->XUnit() )
  {
    g_log.error("Input workspace has not had its unit set");
    throw std::runtime_error("Input workspace has not had its unit set");
  }
  
  // Calculate the number of spectra in this workspace
  const int numberOfSpectra = inputWS->size() / inputWS->blocksize();
  
  API::Workspace_sptr outputWS = getProperty("OutputWorkspace");
  // If input and output workspaces are not the same, create a new workspace for the output
  if (outputWS != inputWS )
  {
    outputWS = WorkspaceFactory::Instance().create(inputWS);
    setProperty("OutputWorkspace",outputWS);
  }
  // Set the final unit that our output workspace will have
  outputWS->XUnit() = UnitFactory::Instance().create(getPropertyValue("Target"));
  
  // Check whether the Y data of the input WS is dimensioned and set output WS flag to be same
  const bool distribution = outputWS->isDistribution(inputWS->isDistribution());
  const unsigned int size = inputWS->blocksize();
  
  // Loop over the histograms (detector spectra)
  for (int i = 0; i < numberOfSpectra; ++i) {
    
    // Take the bin width dependency out of the Y & E data
    if (distribution)
    {
      for (unsigned int j = 0; j < size; ++j)
      {
        const double width = std::abs( inputWS->dataX(i)[j+1] - inputWS->dataX(i)[j] );
        outputWS->dataY(i)[j] = inputWS->dataY(i)[j]*width;
        outputWS->dataE(i)[j] = inputWS->dataE(i)[j]*width;
      }
    }
    else
    {
      // Just copy over
      outputWS->dataY(i) = inputWS->dataY(i);
      outputWS->dataE(i) = inputWS->dataE(i); 
      /// @todo Will also need to deal with E2
    }
    // Copy over the X data (no copying will happen if the two workspaces are the same)
    outputWS->dataX(i) = inputWS->dataX(i);
    
  }  
  
  // Check whether there is a quick conversion available
  double factor, power;
  if ( inputWS->XUnit()->quickConversion(*outputWS->XUnit(),factor,power) )
  // If test fails, could also check whether a quick conversion in the opposite direction has been entered
  {
    convertQuickly(numberOfSpectra,outputWS,factor,power);
  }
  else
  {
    convertViaTOF(numberOfSpectra,inputWS,outputWS);
  }
  
  // If appropriate, put back the bin width division into Y/E.
  if (distribution)
  {
    for (int i = 0; i < numberOfSpectra; ++i) {
      // There must be good case for having a 'divideByBinWidth'/'normalise' algorithm...
      for (unsigned int j = 0; j < size; ++j)
      {
        const double width = std::abs( outputWS->dataX(i)[j+1] - outputWS->dataX(i)[j] );
        outputWS->dataY(i)[j] = inputWS->dataY(i)[j]/width;
        outputWS->dataE(i)[j] = inputWS->dataE(i)[j]/width;
        // Again, will also need to deal with E2
      }
    }
  }
  
}

/// Convert the workspace units according to a simple output = a * (input^b) relationship
void ConvertUnits::convertQuickly(const int& numberOfSpectra, API::Workspace_sptr outputWS, const double& factor, const double& power)
{
  // Loop over the histograms (detector spectra)
  for (int i = 0; i < numberOfSpectra; ++i) {
    std::vector<double>::iterator it;
    for (it = outputWS->dataX(i).begin(); it != outputWS->dataX(i).end(); ++it)
    {
      *it = factor * std::pow(*it,power);
    }
  }
}

/// Convert the workspace units using TOF as an intermediate step in the conversion
void ConvertUnits::convertViaTOF(const int& numberOfSpectra, API::Workspace_sptr inputWS, API::Workspace_sptr outputWS)
{  
  // Get a reference to the instrument contained in the workspace
  boost::shared_ptr<API::Instrument> instrument = inputWS->getInstrument();
  
  // Get the distance between the source and the sample (assume in metres)
  Geometry::ObjComponent* samplePos = instrument->getSamplePos();
  double l1;
  try 
  {
    l1 = instrument->getSource()->getDistance(*samplePos);
    g_log.debug() << "Source-sample distance: " << l1 << std::endl;
  } 
  catch (Exception::NotFoundError e) 
  {
    g_log.error("Unable to calculate source-sample distance");
    throw Exception::InstrumentDefinitionError("Unable to calculate source-sample distance", inputWS->getTitle());
  }
  
  const int notFailed = -99;
  int failedDetectorIndex = notFailed;

  // Not doing anything with the Y vector in to/fromTOF yet, so just pass empty vector
  std::vector<double> emptyVec;
  
  // Loop over the histograms (detector spectra)
  for (int i = 0; i < numberOfSpectra; ++i) {
    
    /// @todo No implementation for any of these in the geometry yet so using properties
    const int emode = getProperty("Emode");
    const double efixed = getProperty("Efixed");
    /// @todo Don't yet consider hold-off (delta)
    const double delta = 0.0;
    
    try {
      // The sample-detector distance for this detector (in metres)
      double l2;
      // The scattering angle for this detector (in radians).
      //     - this assumes the incident beam comes in along the z axis
      double twoTheta;
      // Get these two values
      instrument->detectorLocation(inputWS->spectraNo(i),l2,twoTheta);
      if (failedDetectorIndex != notFailed)
      {
        g_log.information() << "Unable to calculate sample-detector[" << failedDetectorIndex << "-" << i-1 << "] distance. Zeroing spectrum." << std::endl;
        failedDetectorIndex = notFailed;
      }
      
      // Convert the input unit to time-of-flight
      inputWS->XUnit()->toTOF(outputWS->dataX(i),emptyVec,l1,l2,twoTheta,emode,efixed,delta);
      // Convert from time-of-flight to the desired unit
      outputWS->XUnit()->fromTOF(outputWS->dataX(i),emptyVec,l1,l2,twoTheta,emode,efixed,delta);

   } catch (Exception::NotFoundError e) {
      // Get to here if exception thrown when calculating distance to detector
      if (failedDetectorIndex == notFailed)
      {
        failedDetectorIndex = i;
      }
      outputWS->dataX(i).assign(outputWS->dataX(i).size(),0.0);
      outputWS->dataY(i).assign(outputWS->dataY(i).size(),0.0);
      outputWS->dataE(i).assign(outputWS->dataE(i).size(),0.0);
    }

  } // loop over spectra
  
  if (failedDetectorIndex != notFailed)
  {
    g_log.information() << "Unable to calculate sample-detector[" << failedDetectorIndex << "-" << numberOfSpectra-1 << "] distance. Zeroing spectrum." << std::endl;
  }
  
}

} // namespace Algorithm
} // namespace Mantid
