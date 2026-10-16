#include "utils/fmt/json.h"
#include <doctest/doctest.h>

struct MyExampleStruct {};
nlohmann::json format_as(MyExampleStruct const &) {
  nlohmann::json j = {
      {"a", 5},
  };

  return j;
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt implementation for json") {
    SUBCASE("directly formatting json") {
      nlohmann::json j = {
          {"a", 5},
          {"bbb", "hello"},
          {"cc", std::vector<int>{5, 2, 3}},
      };

      std::string result = fmt::to_string(j);
      std::string correct = "{\"a\":5,\"bbb\":\"hello\",\"cc\":[5,2,3]}";

      CHECK(result == correct);
    }

    SUBCASE("using format_as to json") {
      MyExampleStruct my_struct;
      std::string result = fmt::to_string(my_struct);
      std::string correct = "{\"a\":5}";

      CHECK(result == correct);
    }
  }
}
