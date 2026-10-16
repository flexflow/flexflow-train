#include "utils/orthotope/bounded_component.h"
#include "test/utils/rapidcheck/doctest.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BoundedComponent constructor") {
    positive_int bound = 5_p;

    SUBCASE("component < bound") {
      nonnegative_int component = 2_n;

      // check that does not throw
      BoundedComponent{component, bound};
    }

    SUBCASE("component + 1 = bound") {
      nonnegative_int component = 4_n;

      // check that does not throw
      BoundedComponent{component, bound};
    }

    SUBCASE("component = bound") {
      nonnegative_int component = 5_n;

      CHECK_THROWS(BoundedComponent{component, bound});
    }

    SUBCASE("component > bound") {
      nonnegative_int component = 8_n;

      CHECK_THROWS(BoundedComponent{component, bound});
    }
  }

  TEST_CASE("BoundedComponent json serialization and deserialization") {
    RC_SUBCASE([](BoundedComponent const &original) -> bool {
      nlohmann::json serialized = original;
      BoundedComponent deserialized = serialized.get<BoundedComponent>();

      return deserialized == original;
    });
  }

  TEST_CASE("Arbitrary<BoundedComponent>") {
    RC_SUBCASE([](BoundedComponent const &) {});
  }
}
