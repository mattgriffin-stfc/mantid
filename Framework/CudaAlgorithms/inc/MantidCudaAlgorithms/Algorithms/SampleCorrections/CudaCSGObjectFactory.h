// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "MantidCudaAlgorithms/Kernel/CudaV3D.h"

namespace Mantid {
namespace Geometry {
class IObject;
class CSGObject;

namespace detail {
class ShapeInfo;
}
}

namespace CudaAlgorithms {
class CudaRule;
class CudaSurface;
class CudaMaterial;
class CudaShapeInfo;
class CudaV3D;
class CudaCSGObject;

/**
 * GenericShape struct, decomposition of the various implementations of the
 * Surface interface. Used for transferring decomposition to the device so that
 * the inheritance hierarchy can be reconstructed in device memory.
 */
struct GenericShape {
  int shape = -1;
  CudaV3D v3d1;
  CudaV3D v3d2;
  double d1;
};

/**
 * GenericRule struct, decomposition of the various implementations of the Rule
 * interface. Used for transferring decomposition to the device so that the
 * inheritance hierarchy can be reconstructed in device memory.
 */
struct GenericRule {
  int rule = -1;
  int childR = -1;
  int childL = -1;
  int sign;
  int surf;
};

/**
 * CudaCSGObjectFactory, facilitates the creation of CudaCSGObject objects in
 * device memory from instantiations of the IObject interface.
 */
class CudaCSGObjectFactory {
public:
  /**
   * Asynchronous method to create a CudaCSGObject from an IObject instantiation
   * @param d_object pre allocated device CudaCSGObject pointer
   * @param h_object the host object to convert
   * @param stream to perform CUDA operations on (default: 0)
   */
  void createCSGObject(CudaCSGObject * d_object,
                       const Geometry::IObject &h_object,
                       const cudaStream_t &stream = 0);

  /**
   * Deconstructor for CudaCSGObjectFactory, tidies up device memory allocated
   * to CudaCSGObject instances.
   */
  ~CudaCSGObjectFactory();

private:
  /**
   * ShapePointer struct, holds the member pointers to device memory for a
   * CudaCSGObject.
   */
  struct ShapePointer {
    CudaMaterial * material;
    CudaShapeInfo * shapeInfo;
    CudaSurface ** surfaces;
    CudaRule ** topRule;
    CudaV3D * shapeInfoPoints;
    GenericShape * genericShapes;
    GenericRule * genericTopRules;
  };

  void buildPoints(ShapePointer &shapePtr,
                   const Geometry::detail::ShapeInfo &shapeInfo);

  void createSurfaceAndRules(ShapePointer &shapePtr,
                             const Geometry::CSGObject &shape,
                             const cudaStream_t &stream = 0);

  /// the shape pointers allocated in device memory which need to be freed
  std::vector<ShapePointer> m_shapePtrs;
};

}
}
