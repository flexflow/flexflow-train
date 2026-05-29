#include "utils/bidict/algorithms/bidict_filtrans_keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict_filtrans_keys") {
    bidict<int, std::string> dict = {
        {1, "one"},
        {2, "two"},
    };

    bidict<std::string, std::string> result =
        bidict_filtrans_keys(dict, [](int k) -> std::optional<std::string> {
          if (k == 1) {
            return std::nullopt;
          } else {
            std::ostringstream oss;
            oss << (k + 1);
            return oss.str();
          }
        });

    bidict<std::string, std::string> correct = {
        {"3", "two"},
    };

    CHECK(result == correct);
  }
}
