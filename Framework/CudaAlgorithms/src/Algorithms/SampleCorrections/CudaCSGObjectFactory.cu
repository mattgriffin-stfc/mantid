// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidCudaAlgorithms/Algorithms/SampleCorrections/CudaCSGObjectFactory.h"

#include <iostream>

#include "MantidAPI/MatrixWorkspace.h"
#include "MantidCudaAlgorithms/Geometry/CudaCone.h"
#include "MantidCudaAlgorithms/Geometry/CudaCSGObject.h"
#include "MantidCudaAlgorithms/Geometry/CudaCylinder.h"
#include "MantidCudaAlgorithms/Geometry/CudaPlane.h"
#include "MantidCudaAlgorithms/Geometry/CudaQuadratic.h"
#include "MantidCudaAlgorithms/Geometry/CudaRules.h"
#include "MantidCudaAlgorithms/Geometry/CudaSphere.h"
#include "MantidCudaAlgorithms/Geometry/CudaShapeInfo.h"
#include "MantidCudaAlgorithms/Kernel/CudaMaterial.h"
#include "MantidCudaAlgorithms/CudaGuard.h"
#include "MantidGeometry/Objects/CSGObject.h"
#include "MantidGeometry/Objects/Rules.h"
#include "MantidGeometry/Instrument/SampleEnvironment.h"
#include "MantidGeometry/Surfaces/Cone.h"
#include "MantidGeometry/Surfaces/Cylinder.h"
#include "MantidGeometry/Surfaces/Plane.h"
#include "MantidGeometry/Surfaces/Sphere.h"


namespace Mantid {

using namespace Geometry;
using namespace Kernel;

namespace CudaAlgorithms {

__global__
void buildRuleStructure(CudaRule ** d_topRules, CudaSurface ** d_surfaces,
                        GenericRule * d_tempTopRules, int topRulessize,
                        GenericShape * d_GenericShape, int tempShapeSize) {

  for (int i = 0; i < tempShapeSize; i++) {
    GenericShape &tst = d_GenericShape[i];

    if (tst.shape == 0) {
      d_surfaces[i] = new CudaPlane(tst.v3d1, tst.d1);
    } else if (tst.shape == 1) {
      d_surfaces[i] = new CudaCylinder(tst.v3d1, tst.v3d2, tst.d1);
    } else if (tst.shape == 2) {
      d_surfaces[i] = new CudaSphere(tst.v3d1, tst.d1);
    } else if (tst.shape == 3) {
      d_surfaces[i] = new CudaCone(tst.v3d1, tst.v3d2, tst.d1);
    } else {
      assert(0);
    }
  }

  for (int i = topRulessize - 1; i >= 0; i--) {
    GenericRule &trt = d_tempTopRules[i];

    if (trt.rule == 0) {
      d_topRules[i] = new CudaSurfPoint(d_surfaces[trt.surf], trt.sign);
    } else if (trt.rule == 1) {
      d_topRules[i] = new CudaIntersection(d_topRules[trt.childL],
              d_topRules[trt.childR]);
    } else if (trt.rule == 2) {
      d_topRules[i] = new CudaUnion(d_topRules[trt.childL],
              d_topRules[trt.childR]);
    } else {
      assert(0);
    }
  }
}

int decomposeRules(std::vector<GenericRule> &h_tempTopRules,
                   const Rule * h_topRule,
                   const std::vector<const Surface *> &surfaces) {

  int insertPoint = h_tempTopRules.size();

  struct GenericRule t = {};
  h_tempTopRules.push_back(t);

  if (h_topRule->className() == "SurfPoint") {
    const Mantid::SurfPoint * rule =
            dynamic_cast<const Mantid::SurfPoint*>(h_topRule);
    h_tempTopRules[insertPoint].rule = 0;

    int index = -1;
    // Iterate over all elements in Vector
    for (size_t i = 0; i < surfaces.size(); i++) {
      if (rule->getKey() == surfaces[i]) {
        index = i;
        break;
      }
    }
    if (index == -1) {
      throw std::runtime_error("Could not find surface for rule");
    }

    h_tempTopRules[insertPoint].surf = index;
    h_tempTopRules[insertPoint].sign = rule->getSign();
  } else if (h_topRule->className() == "Intersection") {
    h_tempTopRules[insertPoint].rule = 1;

    int childL = decomposeRules(h_tempTopRules, h_topRule->leaf(0), surfaces);
    h_tempTopRules[insertPoint].childL = childL;

    int childR = decomposeRules(h_tempTopRules, h_topRule->leaf(1), surfaces);
    h_tempTopRules[insertPoint].childR = childR;
  } else if (h_topRule->className() == "Union") {
    h_tempTopRules[insertPoint].rule = 2;

    int childL = decomposeRules(h_tempTopRules, h_topRule->leaf(0), surfaces);
    h_tempTopRules[insertPoint].childL = childL;

    int childR = decomposeRules(h_tempTopRules, h_topRule->leaf(1), surfaces);
    h_tempTopRules[insertPoint].childR = childR;
 } else {
    throw std::runtime_error("Cannot process rule of type: " +
                             h_topRule->className());
 }

  return insertPoint;
}

void CudaCSGObjectFactory::createSurfaceAndRules(
        ShapePointer &shapePtr,
        const CSGObject &shape,
        const cudaStream_t &stream) {

  std::vector<GenericRule> h_tempTopRules;
  std::vector<GenericShape> h_genericShape;

  auto &surfaces = shape.getSurfacePtr();
  const unsigned int numberOfSurfaces = surfaces.size();

  for (unsigned int i = 0; i < numberOfSurfaces; ++i) {
    const Surface * surface = surfaces[i];

    GenericShape tst = {};

    if (surface->className() == "Cylinder") {
      const Cylinder* cylinder = dynamic_cast<const Cylinder*>(surfaces[i]);
      tst.shape = 1;
      tst.v3d1 = cylinder->getCentre();
      tst.v3d2 = cylinder->getNormal();
      tst.d1 = cylinder->getRadius();
    } else if (surface->className() == "Plane") {
      const Plane* plane = dynamic_cast<const Plane*>(surfaces[i]);
      tst.shape = 0;
      tst.d1 = plane->getDistance();
      tst.v3d1 = plane->getNormal();
    } else if (surface->className() == "Sphere") {
      const Sphere* plane = dynamic_cast<const Sphere*>(surfaces[i]);
      tst.shape = 2;
      tst.d1 = plane->getRadius();
      tst.v3d1 = plane->getCentre();
    } else if (surface->className() == "Cone") {
      const Cone* plane = dynamic_cast<const Cone*>(surfaces[i]);
      tst.shape = 3;
      tst.d1 = plane->getCosAngle();
      tst.v3d1 = plane->getCentre();
      tst.v3d2 = plane->getNormal();
    } else {
      throw std::runtime_error("Cannot process surface of type: " +
                               surface->className());
    }

    h_genericShape.emplace_back(tst);
  }

  decomposeRules(h_tempTopRules, shape.topRule(), surfaces);

  // allocate temporary objects and rules
  const size_t ruleSize = sizeof(CudaRule*) * h_tempTopRules.size();
  CUDA_GUARD(cudaMalloc(&shapePtr.topRule, ruleSize));

  const size_t genericShapeSize = sizeof(GenericShape) * numberOfSurfaces;
  CUDA_GUARD(cudaMalloc(&shapePtr.genericShapes, genericShapeSize));
  CUDA_GUARD(cudaMemcpy(shapePtr.genericShapes, &h_genericShape[0],
                        genericShapeSize, cudaMemcpyHostToDevice));

  const size_t genericRuleSize = sizeof(GenericRule) * h_tempTopRules.size();
  CUDA_GUARD(cudaMalloc(&shapePtr.genericTopRules, genericRuleSize));
  CUDA_GUARD(cudaMemcpy(shapePtr.genericTopRules, &h_tempTopRules[0],
                        genericRuleSize, cudaMemcpyHostToDevice));

  // launch setup kernel
  buildRuleStructure<<<1, 1, 0, stream>>>(shapePtr.topRule, shapePtr.surfaces,
                                          shapePtr.genericTopRules,
                                          h_tempTopRules.size(),
                                          shapePtr.genericShapes,
                                          numberOfSurfaces);
}

void CudaCSGObjectFactory::buildPoints(ShapePointer &shapePtr,
                                       const ShapeInfo &shapeInfo) {

  ShapeInfo::GeometryShape shape = shapeInfo.shape();
  std::vector<V3D> points = shapeInfo.points();

  // build rotation matrix
  if (shape == ShapeInfo::GeometryShape::CONE ||
      shape == ShapeInfo::GeometryShape::CYLINDER ||
      shape == ShapeInfo::GeometryShape::HOLLOWCYLINDER) {

    const V3D &axis = points.back();
    points.pop_back();

    // default z aligned axis
    V3D currVec(0, 0, 1);

    const double c = currVec.scalar_prod(axis);
    V3D v = currVec.cross_prod(axis);
    const double l = 1 / 1 + c;

    Matrix<double> identity(3, 3, true);
    Matrix<double> skv(3, 3, true);
    skv[0][0] = 0;
    skv[0][1] = -v[2];
    skv[0][2] = v[1];
    skv[1][0] = v[2];
    skv[1][1] = 0;
    skv[1][2] = -v[0];
    skv[2][0] = -v[1];
    skv[2][1] = v[0];
    skv[2][2] = 0;

    Matrix<double> output = identity + skv + (skv*skv)*l;

    points.emplace_back(output[0][0], output[0][1], output[0][2]); // X
    points.emplace_back(output[1][0], output[1][1], output[1][2]); // Y
    points.emplace_back(output[2][0], output[2][1], output[2][2]); // Z
  }

  std::vector<CudaV3D> cudaPoints;

  for (auto &point : points) {
    cudaPoints.emplace_back(point);
  }

  const size_t pointsSize = sizeof(CudaV3D) * cudaPoints.size();
  CUDA_GUARD(cudaMalloc(&shapePtr.shapeInfoPoints, pointsSize));
  CUDA_GUARD(cudaMemcpy(shapePtr.shapeInfoPoints, &cudaPoints[0], pointsSize,
             cudaMemcpyHostToDevice));
}

void CudaCSGObjectFactory::createCSGObject(CudaCSGObject * d_object,
                                           const IObject &rawShape,
                                           const cudaStream_t &stream) {

  ShapePointer shapePtr;
  // if the object is a container then get its geometry
  auto containerShape = dynamic_cast<const Container*>(&rawShape);

  const IObject * targetShape = &(containerShape ? containerShape->getShape()
                                                 : rawShape);

  // ensure the geometry is CSG
  auto castShape = dynamic_cast<const CSGObject*>(targetShape);
  if (!castShape) {
    std::ostringstream error;
    error << "This algorithm only supports CSG Object shapes. "
          << rawShape.id() << " given."
          << std::endl;
    throw std::runtime_error("This algorithm only supports CSG Object shapes");
  }

  const CSGObject shape = *castShape;

  // copy surfaces and rules
  unsigned int nsurfaces = static_cast<unsigned int>(shape.getSurfacePtr()
                                                     .size());

  CUDA_GUARD(cudaMalloc(&shapePtr.surfaces, sizeof(CudaSurface*) * nsurfaces));
  createSurfaceAndRules(shapePtr, shape, stream);
  // copy material
  const Material &material = shape.material();

  CudaMaterial h_material(material.numberDensity(),
                          material.totalScatterXSection(),
                          material.absorbXSection(1));

  const size_t materialsSize = sizeof(CudaMaterial);

  CUDA_GUARD(cudaMalloc(&shapePtr.material, materialsSize));
  CUDA_GUARD(cudaMemcpyAsync(shapePtr.material, &h_material, materialsSize,
                             cudaMemcpyHostToDevice, stream));

  // copy shape info
  const ShapeInfo &shapeInfo = shape.shapeInfo();
  buildPoints(shapePtr, shapeInfo);

  CudaShapeInfo h_shapeInfo(shapeInfo.height(), shapeInfo.radius(),
                            shapeInfo.innerRadius(), shapeInfo.shape(),
                            shapePtr.shapeInfoPoints);

  const size_t shapeInfoSize = sizeof(CudaShapeInfo);
  CUDA_GUARD(cudaMalloc(&shapePtr.shapeInfo, shapeInfoSize));
  CUDA_GUARD(cudaMemcpyAsync(shapePtr.shapeInfo, &h_shapeInfo, shapeInfoSize,
                             cudaMemcpyHostToDevice, stream));

  // copy csg object
  CudaCSGObject h_object(shapePtr.shapeInfo, shapePtr.material,
                         shapePtr.topRule, shapePtr.surfaces, nsurfaces);

  const size_t objectSize = sizeof(CudaCSGObject);
  CUDA_GUARD(cudaMemcpyAsync(d_object, &h_object, objectSize,
                             cudaMemcpyHostToDevice, stream));

  m_shapePtrs.emplace_back(shapePtr);
}

CudaCSGObjectFactory::~CudaCSGObjectFactory() {
  for (auto &shapePtr : m_shapePtrs) {
    CUDA_SOFT_GUARD(cudaFree(shapePtr.material));
    CUDA_SOFT_GUARD(cudaFree(shapePtr.shapeInfo));
    CUDA_SOFT_GUARD(cudaFree(shapePtr.surfaces));
    CUDA_SOFT_GUARD(cudaFree(shapePtr.topRule));
    CUDA_SOFT_GUARD(cudaFree(shapePtr.shapeInfoPoints));
    CUDA_SOFT_GUARD(cudaFree(shapePtr.genericTopRules));
    CUDA_SOFT_GUARD(cudaFree(shapePtr.genericShapes));
  }
}

}
}
