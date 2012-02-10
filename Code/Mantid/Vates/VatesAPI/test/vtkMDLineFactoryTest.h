#ifndef VTK_MD_LINE_FACTORY_TEST
#define VTK_MD_LINE_FACTORY_TEST

#include <cxxtest/TestSuite.h>
#include "MantidDataObjects/TableWorkspace.h"
#include "MantidVatesAPI/vtkMDLineFactory.h"
#include "MantidVatesAPI/NoThresholdRange.h"
#include "MockObjects.h"
#include "MantidMDEvents/SliceMD.h"
#include "MantidTestHelpers/MDEventsTestHelper.h"
#include "vtkCellType.h"
#include "vtkUnstructuredGrid.h"

using namespace Mantid::VATES;
using namespace Mantid::API;
using namespace Mantid::MDEvents;
using namespace testing;

//=====================================================================================
// Functional tests
//=====================================================================================
class vtkMDLineFactoryTest : public CxxTest::TestSuite
{
public:

  void testGetFactoryTypeName()
  {
    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    TS_ASSERT_EQUALS("vtkMDLineFactory", factory.getFactoryTypeName());
  }

  void testInitializeDelegatesToSuccessor()
  {
    MockvtkDataSetFactory* mockSuccessor = new MockvtkDataSetFactory;
    EXPECT_CALL(*mockSuccessor, initialize(_)).Times(1);
    EXPECT_CALL(*mockSuccessor, getFactoryTypeName()).Times(1);

    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    factory.SetSuccessor(mockSuccessor);

    ITableWorkspace_sptr ws(new Mantid::DataObjects::TableWorkspace);
    TS_ASSERT_THROWS_NOTHING(factory.initialize(ws));

    TSM_ASSERT("Successor has not been used properly.", Mock::VerifyAndClearExpectations(mockSuccessor));
  }

  void testCreateDelegatesToSuccessor()
  {
    MockvtkDataSetFactory* mockSuccessor = new MockvtkDataSetFactory;
    EXPECT_CALL(*mockSuccessor, initialize(_)).Times(1);
    EXPECT_CALL(*mockSuccessor, create()).Times(1);
    EXPECT_CALL(*mockSuccessor, getFactoryTypeName()).Times(1);

    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    factory.SetSuccessor(mockSuccessor);

    ITableWorkspace_sptr ws(new Mantid::DataObjects::TableWorkspace);
    TS_ASSERT_THROWS_NOTHING(factory.initialize(ws));
    TS_ASSERT_THROWS_NOTHING(factory.create());

    TSM_ASSERT("Successor has not been used properly.", Mock::VerifyAndClearExpectations(mockSuccessor));
  }

  void testOnInitaliseCannotDelegateToSuccessor()
  {
    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    //factory.SetSuccessor(mockSuccessor); No Successor set.

    ITableWorkspace_sptr ws(new Mantid::DataObjects::TableWorkspace);
    TS_ASSERT_THROWS(factory.initialize(ws), std::runtime_error);
  }

  void testCreateWithoutInitializeThrows()
  {
    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    //initialize not called!
    TS_ASSERT_THROWS(factory.create(), std::runtime_error);
  }

  void testCreation()
  {
    boost::shared_ptr<Mantid::MDEvents::MDEventWorkspace<Mantid::MDEvents::MDEvent<1>,1> >
            ws = MDEventsTestHelper::makeMDEWFull<1>(10, 10, 10, 10);

    //Rebin it to make it possible to compare cells to bins.
    SliceMD slice;
    slice.initialize();
    slice.setProperty("InputWorkspace", ws);
    slice.setPropertyValue("AlignedDimX", "Axis0, -10, 10, 100");
    slice.setPropertyValue("OutputWorkspace", "binned");
    slice.execute();

    Workspace_sptr binned = Mantid::API::AnalysisDataService::Instance().retrieve("binned");

    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    factory.initialize(binned);

    vtkDataSet* product = factory.create();

    TS_ASSERT(dynamic_cast<vtkUnstructuredGrid*>(product) != NULL);
    TS_ASSERT_EQUALS(100, product->GetNumberOfCells());
    TS_ASSERT_EQUALS(200, product->GetNumberOfPoints());
    TS_ASSERT_EQUALS(VTK_LINE, product->GetCellType(0));

    product->Delete();
    AnalysisDataService::Instance().remove("binned");
  }

};

//=====================================================================================
// Peformance tests
//=====================================================================================
class vtkMDLineFactoryTestPerformance : public CxxTest::TestSuite
{

public:

  void setUp()
  {
    boost::shared_ptr<Mantid::MDEvents::MDEventWorkspace<Mantid::MDEvents::MDEvent<1>,1> > input 
      = MDEventsTestHelper::makeMDEWFull<1>(2, 10, 10, 4000);
    //Rebin it to make it possible to compare cells to bins.
    SliceMD slice;
    slice.initialize();
    slice.setProperty("InputWorkspace", input);
    slice.setPropertyValue("AlignedDimX", "Axis0, -10, 10, 200000");
    slice.setPropertyValue("OutputWorkspace", "binned");
    slice.execute();
  }

  void tearDown()
  {
    AnalysisDataService::Instance().remove("binned");
  }

  void testCreationOnLargeWorkspace()
  {
    Workspace_sptr binned = Mantid::API::AnalysisDataService::Instance().retrieve("binned");

    vtkMDLineFactory factory(ThresholdRange_scptr(new NoThresholdRange), "signal");
    factory.initialize(binned);

    vtkDataSet* product = factory.create();

    TS_ASSERT(dynamic_cast<vtkUnstructuredGrid*>(product) != NULL);
    TS_ASSERT_EQUALS(200000, product->GetNumberOfCells());
    TS_ASSERT_EQUALS(400000, product->GetNumberOfPoints());

    product->Delete();
    
  }
};

#endif
  