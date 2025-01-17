#include "utils/bidict/bidict.h"
#include "test/utils/doctest/check_without_stringify.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict") {
    bidict<int, std::string> dict;
    dict.equate(1, "one");
    dict.equate(2, "two");

    SUBCASE("L type is the same as R type") {
      bidict<int, int> bd;
      bd.equate(1, 3);

      SUBCASE("bidict::contains_l") {
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("bidict::contains_r") {
        CHECK(bd.contains_r(3));
        CHECK_FALSE(bd.contains_r(1));
      }
    }

    SUBCASE("L type is not the same as R type") {
      bidict<int, std::string> dict;
      dict.equate(1, "one");
      dict.equate(2, "two");

      SUBCASE("bidict::contains_l") {
        CHECK(dict.contains_l(1));
        CHECK_FALSE(dict.contains_l(3));
      }

      SUBCASE("bidict::contains_r") {
        CHECK(dict.contains_r("one"));
        CHECK_FALSE(dict.contains_r("three"));
      }
    }

    SUBCASE("bidict::equate") {
      CHECK(dict.at_l(1) == "one");
      CHECK(dict.at_r("one") == 1);
      CHECK(dict.at_l(2) == "two");
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("bidict::erase_l") {
      dict.erase_l(1);
      CHECK(dict.size() == 1);
      CHECK_THROWS_AS(dict.at_l(1), std::out_of_range);
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("bidict::erase_r") {
      dict.erase_r("one");
      CHECK(dict.size() == 1);
      CHECK_THROWS_AS(dict.at_r("one"), std::out_of_range);
      CHECK(dict.at_l(2) == "two");
    }

    SUBCASE("bidict::reversed") {
      bidict<std::string, int> reversed_dict = dict.reversed();
      CHECK(reversed_dict.at_l("one") == 1);
      CHECK(reversed_dict.at_r(2) == "two");
    }

    SUBCASE("bidict::size") {
      CHECK(dict.size() == 2);
    }

    SUBCASE("implicitly convert to std::unordered_map") {
      std::unordered_map<int, std::string> res = dict;
      std::unordered_map<int, std::string> expected = {{1, "one"}, {2, "two"}};
      CHECK(res == expected);
    }

    SUBCASE("bidict::begin") {
      auto it = dict.begin();
      CHECK(it->first == 2);
      CHECK(it->second == "two");
    }

    SUBCASE("bidict::end") {
      auto it = dict.end();

      CHECK_WITHOUT_STRINGIFY(it == dict.end());
    }

    SUBCASE("fmt::to_string(bidict<int, std::string>)") {
      std::string result = fmt::to_string(dict);
      std::string correct = fmt::to_string(dict.as_unordered_map());
      CHECK(result == correct);
    }
  }
}
