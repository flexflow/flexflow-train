#include <doctest/doctest.h>
#include "utils/json/monostate.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("adl_serializer<std::monostate>") {
    SUBCASE("to_json") {
      std::monostate input = std::monostate{};

      nlohmann::json result = input;
      nlohmann::json correct = nullptr;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      SUBCASE("json is nullptr") {
        nlohmann::json input = nullptr;

        std::monostate result = input.get<std::monostate>();
        std::monostate correct = std::monostate{};

        CHECK(result == correct);
      }

      SUBCASE("json is not nullptr") {
        nlohmann::json input = 5;

        CHECK_THROWS(input.get<std::monostate>());
      }
    }
  }
}
