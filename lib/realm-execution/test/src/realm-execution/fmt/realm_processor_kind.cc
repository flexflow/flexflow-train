#include <doctest/doctest.h>
#include "realm-execution/fmt/realm_processor_kind.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(Realm::Processor::Kind)") {
    std::string result = fmt::to_string(::FlexFlow::Realm::Processor::Kind::TOC_PROC);
    std::string correct = "<Processor::Kind::TOC_PROC>";

    CHECK(result == correct);
  }
}
