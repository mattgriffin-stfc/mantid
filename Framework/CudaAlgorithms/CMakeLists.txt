set(SRC_FILES
    src/Algorithms/SampleCorrections/CudaBeamProfileFactory.cu
    src/Algorithms/SampleCorrections/CudaCSGObjectFactory.cu
    src/Algorithms/SampleCorrections/CudaDetectorBinProvider.cu
    src/Algorithms/SampleCorrections/CudaDetectorBinProviderSetup.cpp
    src/Algorithms/SampleCorrections/CudaMCAbsorptionSetup.cpp
    src/Algorithms/SampleCorrections/CudaMCAbsorptionStrategy.cu
    src/Algorithms/SampleCorrections/CudaMCGeometryBuffer.cu
    src/Algorithms/SampleCorrections/CudaRectangularBeamProfile.cu
    src/Algorithms/CudaMonteCarloAbsorption.cpp
    src/Geometry/CudaBoundingBox.cu
    src/Geometry/CudaCone.cu
    src/Geometry/CudaCSGObject.cu
    src/Geometry/CudaCylinder.cu
    src/Geometry/CudaLine.cu
    src/Geometry/CudaLineIntersectVisit.cu
    src/Geometry/CudaPlane.cu
    src/Geometry/CudaQuadratic.cu
    src/Geometry/CudaRandomPoint.cu
    src/Geometry/CudaRuleItems.cu
    src/Geometry/CudaSphere.cu
    src/Geometry/CudaShapeInfo.cu
    src/Geometry/CudaTrack.cu
    src/Kernel/CudaMaterial.cu
    src/Kernel/CudaMersenneTwister.cu
    src/Kernel/CudaV3D.cu
    src/CudaAlgorithmContext.cu
    src/CudaMathExtensions.cu
    src/CudaVector.cu)

set(INC_FILES
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaBeamProfileFactory.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaCSGObjectFactory.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBin.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProvider.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaDetectorBinProviderSetup.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionParameters.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionSetup.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCAbsorptionStrategy.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaMCGeometryBuffer.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaRectangularBeamProfile.h
    inc/MantidCudaAlgorithms/Algorithms/SampleCorrections/ICudaBeamProfile.h
    inc/MantidCudaAlgorithms/Algorithms/CudaMonteCarloAbsorption.h
    inc/MantidCudaAlgorithms/Geometry/CudaBoundingBox.h
    inc/MantidCudaAlgorithms/Geometry/CudaCone.h
    inc/MantidCudaAlgorithms/Geometry/CudaCSGObject.h
    inc/MantidCudaAlgorithms/Geometry/CudaCylinder.h
    inc/MantidCudaAlgorithms/Geometry/CudaLine.h
    inc/MantidCudaAlgorithms/Geometry/CudaLineIntersectVisit.h
    inc/MantidCudaAlgorithms/Geometry/CudaPlane.h
    inc/MantidCudaAlgorithms/Geometry/CudaQuadratic.h
    inc/MantidCudaAlgorithms/Geometry/CudaRandomPoint.h
    inc/MantidCudaAlgorithms/Geometry/CudaRules.h
    inc/MantidCudaAlgorithms/Geometry/CudaShapeInfo.h
    inc/MantidCudaAlgorithms/Geometry/CudaSphere.h
    inc/MantidCudaAlgorithms/Geometry/CudaSurface.h
    inc/MantidCudaAlgorithms/Geometry/CudaTrack.h
    inc/MantidCudaAlgorithms/Kernel/CudaMaterial.h
    inc/MantidCudaAlgorithms/Kernel/CudaMersenneTwister.h
    inc/MantidCudaAlgorithms/Kernel/CudaV3D.h
    inc/MantidCudaAlgorithms/CudaAlgorithmContext.h
    inc/MantidCudaAlgorithms/CudaGuard.h
    inc/MantidCudaAlgorithms/CudaMathExtensions.h
    inc/MantidCudaAlgorithms/CurandGuard.h
    inc/MantidCudaAlgorithms/CudaReduce.h
    inc/MantidCudaAlgorithms/CudaVector.h)

if(COVERALLS)
  foreach(loop_var ${SRC_FILES} ${C_SRC_FILES} ${INC_FILES})
    set_property(GLOBAL APPEND
                 PROPERTY COVERAGE_SRCS
                          "${CMAKE_CURRENT_SOURCE_DIR}/${loop_var}")
  endforeach(loop_var)
endif()

# Add a precompiled header where they are supported
#enable_precompiled_headers(inc/MantidCudaAlgorithms/PrecompiledHeader.h SRC_FILES)

add_library(CudaAlgorithms ${SRC_FILES} ${C_SRC_FILES} ${INC_FILES})

# Set the name of the generated library
set_target_properties(CudaAlgorithms
                      PROPERTIES OUTPUT_NAME
                                 MantidCudaAlgorithms
                                 COMPILE_DEFINITIONS
                                 "IN_MANTID_ALGORITHMS"
                                 CUDA_SEPARABLE_COMPILATION
                                 ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_target_properties(CudaAlgorithms
                        PROPERTIES INSTALL_RPATH
                                   "@loader_path/../Contents/MacOS")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set_target_properties(CudaAlgorithms
                        PROPERTIES INSTALL_RPATH "\$ORIGIN/../${LIB_DIR}")
endif()

# Add to the 'Framework' group in VS
set_property(TARGET CudaAlgorithms PROPERTY FOLDER "MantidFramework")

include_directories(inc)

target_include_directories(CudaAlgorithms SYSTEM PUBLIC ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
    /home/matt/Documents/stfc/mantid/Framework/Algorithms/inc/) # Temporary
target_link_libraries(CudaAlgorithms
                      LINK_PRIVATE
                      ${TCMALLOC_LIBRARIES_LINKTIME}
                      ${MANTIDLIBS}
                      ${GSL_LIBRARIES}
                      ${CUDA_LIBRARIES})

# Auto-generate exports header
target_include_directories(CudaAlgorithms PUBLIC ${CMAKE_BINARY_DIR}/Framework/CudaAlgorithms)
generate_mantid_export_header(CudaAlgorithms FALSE)

# Installation settings

mtd_install_targets(TARGETS
                    CudaAlgorithms
                    INSTALL_DIRS
                    ${PLUGINS_DIR}
                    ${WORKBENCH_PLUGINS_DIR})
