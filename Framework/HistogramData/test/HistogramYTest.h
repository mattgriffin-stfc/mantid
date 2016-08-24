#ifndef MANTID_HISTOGRAMDATA_HISTOGRAMYTEST_H_
#define MANTID_HISTOGRAMDATA_HISTOGRAMYTEST_H_

#include <cxxtest/TestSuite.h>

#include "MantidHistogramData/Addable.h"
#include "MantidHistogramData/FixedLengthVector.h"
#include "MantidHistogramData/HistogramY.h"
#include "MantidHistogramData/Offsetable.h"
#include "MantidHistogramData/Scalable.h"

using namespace Mantid;
using namespace HistogramData;

class HistogramYTest : public CxxTest::TestSuite {
public:
  // This pair of boilerplate methods prevent the suite being created statically
  // This means the constructor isn't called when running other tests
  static HistogramYTest *createSuite() { return new HistogramYTest(); }
  static void destroySuite(HistogramYTest *suite) { delete suite; }

  // The implementation of the testee is based on mixins and a few
  // compiler-generated constructors. Tests for the features provided by mixins
  // can be found in the respective mixin tests. Here we simply test that we
  // have the expected mixins.

  void test_has_correct_mixins() {
    HistogramY y;
// AppleClang gives warning if the result is unused.
#if __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#endif
    TS_ASSERT_THROWS_NOTHING(
        dynamic_cast<detail::FixedLengthVector<HistogramY> &>(y));
    TS_ASSERT_THROWS_NOTHING(dynamic_cast<detail::Addable<HistogramY> &>(y));
    TS_ASSERT_THROWS_NOTHING(dynamic_cast<detail::Offsetable<HistogramY> &>(y));
    TS_ASSERT_THROWS_NOTHING(dynamic_cast<detail::Scalable<HistogramY> &>(y));
#if __clang__
#pragma clang diagnostic pop
#endif
  }
};

#endif /* MANTID_HISTOGRAMDATA_HISTOGRAMYTEST_H_ */