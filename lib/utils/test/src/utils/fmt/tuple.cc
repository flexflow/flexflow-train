#include "utils/fmt/tuple.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::tuple)") {
    SUBCASE("types are different") {
      std::tuple<int, bool, std::string> input = {3, false, "hello"};

      std::string result = fmt::to_string(input);
      std::string correct = "{3, false, hello}";

      CHECK(result == correct);
    }

    SUBCASE("types are the same") {
      std::tuple<int, int> input = {3, 5};

      std::string result = fmt::to_string(input);
      std::string correct = "{3, 5}";

      CHECK(result == correct);
    }

    SUBCASE("empty tuple") {
      std::tuple<> input = {};

      std::string result = fmt::to_string(input);
      std::string correct = "{}";

      CHECK(result == correct);
    }
  }

  TEST_CASE("operator<<(ostream &, std::tuple)") {
    auto through_ostringstream = [](auto const &t) {
      std::ostringstream oss;
      oss << t;
      return oss.str();
    };

    SUBCASE("types are different") {
      std::tuple<int, bool, std::string> input = {3, false, "hello"};

      std::string result = through_ostringstream(input);
      std::string correct = "{3, false, hello}";

      CHECK(result == correct);
    }

    SUBCASE("types are the same") {
      std::tuple<int, int> input = {3, 5};

      std::string result = through_ostringstream(input);
      std::string correct = "{3, 5}";

      CHECK(result == correct);
    }

    SUBCASE("empty tuple") {
      std::tuple<> input = {};

      std::string result = through_ostringstream(input);
      std::string correct = "{}";

      CHECK(result == correct);
    }
  }
}
