#ifndef MANTID_KERNEL_ENABLEDWHENPROPERTYTEST_H_
#define MANTID_KERNEL_ENABLEDWHENPROPERTYTEST_H_

#include <cxxtest/TestSuite.h>
#include "MantidKernel/Timer.h"
#include "MantidKernel/System.h"

#include "MantidKernel/EnabledWhenProperty.h"
#include "MantidKernel/PropertyManager.h"
#include "MantidKernel/PropertyManagerOwner.h"
#include "MantidKernel/Property.h"

using namespace Mantid;
using namespace Mantid::Kernel;

class EnabledWhenPropertyTest : public CxxTest::TestSuite {
public:
  void test_when_IS_NOT_DEFAULT() {
    PropertyManagerOwner alg;
    // Start with a regular property
    alg.declareProperty("DependantProp", 123);

    // Make a property with its validator. Will be enabled when that other one
    // is NOT the default
    auto val = [] {
      return make_unique<EnabledWhenProperty>("DependantProp", IS_NOT_DEFAULT);
    };
    alg.declareProperty("EnabledProp", 456);
    alg.setPropertySettings("EnabledProp", val());

    Property *prop = alg.getPointerToProperty("EnabledProp");
    TS_ASSERT(prop);
    if (!prop)
      return;
    TSM_ASSERT("Property always returns visible.",
               prop->getSettings()->isVisible(&alg))
    TSM_ASSERT("Property always returns valid.", prop->isValid().empty())

    TSM_ASSERT("Starts off NOT enabled", !prop->getSettings()->isEnabled(&alg));
    alg.setProperty("DependantProp", 234);
    TSM_ASSERT("Becomes enabled when another property has been changed",
               prop->getSettings()->isEnabled(&alg));

    alg.declareProperty("MySecondValidatorProp", 456);
    alg.setPropertySettings("MySecondValidatorProp", val());
    prop = alg.getPointerToProperty("MySecondValidatorProp");
    TSM_ASSERT("Starts off enabled", prop->getSettings()->isEnabled(&alg));
    alg.setProperty("DependantProp", 123);
    TSM_ASSERT("Goes back to disabled", !prop->getSettings()->isEnabled(&alg));
  }

  void test_when_IS_DEFAULT() {
    PropertyManagerOwner alg;
    alg.declareProperty("DependantProp", 123);
    // Make a property with its validator. Will be enabled when that other one
    // is the default
    alg.declareProperty("EnabledProp", 456);
    alg.setPropertySettings("EnabledProp", make_unique<EnabledWhenProperty>(
                                               "DependantProp", IS_DEFAULT));
    Property *prop = alg.getPointerToProperty("EnabledProp");
    TS_ASSERT(prop);
    if (!prop)
      return;
    TSM_ASSERT("Starts off enabled", prop->getSettings()->isEnabled(&alg));
    alg.setProperty("DependantProp", -1);
    TSM_ASSERT("Becomes disabled when another property has been changed",
               !prop->getSettings()->isEnabled(&alg));
  }

  void test_when_IS_EQUAL_TO() {
    PropertyManagerOwner alg;
    alg.declareProperty("DependantProp", 123);
    alg.declareProperty("EnabledProp", 456);
    alg.setPropertySettings(
        "EnabledProp",
        make_unique<EnabledWhenProperty>("DependantProp", IS_EQUAL_TO, "234"));
    Property *prop = alg.getPointerToProperty("EnabledProp");
    TS_ASSERT(prop);
    if (!prop)
      return;
    TSM_ASSERT("Starts off disabled", !prop->getSettings()->isEnabled(&alg));
    alg.setProperty("DependantProp", 234);
    TSM_ASSERT(
        "Becomes enabled when the other property is equal to the given string",
        prop->getSettings()->isEnabled(&alg));
  }

  void test_when_IS_NOT_EQUAL_TO() {
    PropertyManagerOwner alg;
    alg.declareProperty("DependantProp", 123);
    alg.declareProperty("EnabledProp", 456);
    alg.setPropertySettings("EnabledProp",
                            make_unique<EnabledWhenProperty>(
                                "DependantProp", IS_NOT_EQUAL_TO, "234"));
    Property *prop = alg.getPointerToProperty("EnabledProp");
    TS_ASSERT(prop);
    if (!prop)
      return;
    TSM_ASSERT("Starts off enabled", prop->getSettings()->isEnabled(&alg));
    alg.setProperty("DependantProp", 234);
    TSM_ASSERT(
        "Becomes disabled when the other property is equal to the given string",
        !prop->getSettings()->isEnabled(&alg));
  }

  void test_combination_AND() {
    // Setup with same value first
    auto alg = setupCombinationTest(AND, true);
    auto prop = alg.getPointerToProperty(m_resultPropName);
    TS_ASSERT(prop);
    // AND should return true first
    TS_ASSERT(prop->getSettings()->isEnabled(&alg));

    // Now set a different value - should be disabled
    alg.setPropertyValue(m_propertyTwoName, m_differentValue);
    TS_ASSERT(!prop->getSettings()->isEnabled(&alg));
  }

private:
  const std::string m_propertyOneValue = "testTrue";
  const std::string m_differentValue = "testFalse";
  const std::string m_resultValue = "Result";

  const std::string m_propertyOneName = "PropOne";
  const std::string m_propertyTwoName = "PropTwo";
  const std::string m_resultPropName = "ResultProp";

  PropertyManagerOwner setupCombinationTest(eLogicOperator logicOperation,
                                            bool isSameValue) {
    auto propOne =
        getEnabledWhenProp(m_propertyOneName, IS_EQUAL_TO, m_propertyOneValue);
    auto propTwo =
        getEnabledWhenProp(m_propertyTwoName, IS_EQUAL_TO, m_propertyOneValue);
    auto combination = getCombinationProperty(
        std::move(propOne), std::move(propTwo), logicOperation);
    // Set both to the same value to check
    PropertyManagerOwner alg;
    alg.declareProperty(m_propertyOneName, m_propertyOneValue);
    if (isSameValue) {
      alg.declareProperty(m_propertyTwoName, m_propertyOneValue);
    } else {
      alg.declareProperty(m_propertyTwoName, m_differentValue);
    }
    alg.declareProperty(m_resultPropName, m_resultValue);
    alg.setPropertySettings(m_resultPropName, std::move(combination));
    return alg;
  }

  std::unique_ptr<EnabledWhenProperty>
  getEnabledWhenProp(const std::string &propName, ePropertyCriterion criterion,
                     const std::string &value = "") {
    if (value.length() == 0) {
      return std::make_unique<EnabledWhenProperty>(propName, criterion);
    } else {
      return std::make_unique<EnabledWhenProperty>(propName, criterion, value);
    }
  }

  using EnabledPropPtr = std::unique_ptr<EnabledWhenProperty>;
  std::unique_ptr<IPropertySettings>
  getCombinationProperty(EnabledPropPtr &&condOne, EnabledPropPtr &&condTwo,
                         eLogicOperator logicalOperator) {
    return std::make_unique<EnabledWhenProperty>(
        std::move(condOne), std::move(condTwo), logicalOperator);
  }
};

#endif /* MANTID_KERNEL_ENABLEDWHENPROPERTYTEST_H_ */
