#include "utils/containers/get_element_type.h"
#include <doctest/doctest.h>
#include <realm.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_element_type_t") {
    SUBCASE("std::vector<int>") {
      bool result = std::is_same_v<get_element_type_t<std::vector<int>>, int>;
      CHECK(result);
    }

    SUBCASE("Realm::Machine::MemoryQuery") {
      bool result =
          std::is_same_v<get_element_type_t<Realm::Machine::MemoryQuery>,
                         Realm::Memory>;
      CHECK(result);
    }

    SUBCASE("Realm::Machine::ProcessorQuery") {
      bool result =
          std::is_same_v<get_element_type_t<Realm::Machine::MemoryQuery>,
                         Realm::Memory>;
      CHECK(result);
    }
  }
}
