#include "utils/tuple/visit.h"
#include "utils/overload.h"
#include <doctest/doctest.h>
#include <sstream>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("visit(std::tuple, Visitor)") {
    std::ostringstream oss;
    auto visitor = overload{
        [&](int const &i) -> void { oss << "int(" << i << "), "; },
        [&](bool const &b) -> void { oss << "bool(" << b << "), "; },
        [&](std::string const &s) -> void { oss << "string(" << s << "), "; },
    };

    SUBCASE("repeated types") {
      std::tuple<int, std::string, bool, std::string> input = {
          3, "hello", false, "world"};

      visit_tuple(input, visitor);

      std::string result = oss.str();
      std::string correct = "int(3), string(hello), bool(0), string(world), ";

      CHECK(result == correct);
    }

    SUBCASE("empty tuple") {
      std::tuple<> input = {};

      visit_tuple(input, visitor);

      std::string result = oss.str();
      std::string correct = "";

      CHECK(result == correct);
    }
  }
}
