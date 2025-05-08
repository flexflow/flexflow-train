#include <doctest/doctest.h>
#include "task-spec/arg_ref.h"
#include <string>

using namespace ::FlexFlow;

enum class ExampleLabelType { 
  STRING,
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ArgRefSpec::holds") {
    CHECK_MESSAGE(false, "TODO: ArgRefSpec");

    ArgRefSpec<ExampleLabelType> arg_ref_spec = ArgRefSpec<ExampleLabelType>::create(
      ArgRef<ExampleLabelType, std::string>{ExampleLabelType::STRING}
    );

    SUBCASE("returns true if the type matches the ArgRef type") {
      bool result = arg_ref_spec.holds<std::string>();
      bool correct = true; 

      CHECK(result == correct);
    }

    SUBCASE("returns false otherwise") {
      bool result = arg_ref_spec.holds<int>();
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
