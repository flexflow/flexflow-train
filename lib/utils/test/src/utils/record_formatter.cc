#include "utils/record_formatter.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

std::string formatRecord(RecordFormatter const &formatter) {
  std::ostringstream oss;
  oss << formatter;
  return oss.str();
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RecordFormatter") {
    RecordFormatter formatter = mk_empty_record(Orientation::HORIZONTAL);

    SUBCASE("Appending string") {
      formatter << "Hello";
      formatter << "World";
      CHECK(formatRecord(formatter) == "{ Hello | World }");
    }

    SUBCASE("Appending integer and float") {
      formatter << 42;
      formatter << 3.14f;
      CHECK(formatRecord(formatter) == "{ 42 | 3.140000e+00 }");
    }

    SUBCASE("Appending another RecordFormatter") {
      RecordFormatter subFormatter = mk_empty_record(Orientation::VERTICAL);
      subFormatter << "Sub";
      subFormatter << "Formatter";

      formatter << "Hello";
      formatter << subFormatter;

      std::ostringstream oss;
      oss << formatter;

      CHECK(formatRecord(formatter) == "{ Hello | { Sub | Formatter } }");
    }
  }
}
