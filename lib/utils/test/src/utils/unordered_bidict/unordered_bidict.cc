#include "utils/unordered_bidict/unordered_bidict.h"
#include "test/utils/doctest/check_without_stringify.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict") {
    unordered_bidict<int, std::string> dict;
    dict.equate(1, "one");
    dict.equate(2, "two");

    SUBCASE("L type is the same as R type") {
      unordered_bidict<int, int> bd;
      bd.equate(1, 3);

      SUBCASE("unordered_bidict::contains_l") {
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("unordered_bidict::contains_r") {
        CHECK(bd.contains_r(3));
        CHECK_FALSE(bd.contains_r(1));
      }
    }

    SUBCASE("L type is not the same as R type") {
      unordered_bidict<int, std::string> bd;
      bd.equate(1, "one");
      bd.equate(2, "two");

      SUBCASE("unordered_bidict::contains_l") {
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("unordered_bidict::contains_r") {
        CHECK(bd.contains_r("one"));
        CHECK_FALSE(bd.contains_r("three"));
      }
    }

    SUBCASE("unordered_bidict::unordered_bidict(std::initializer_list<std::"
            "pair<L, R>>)") {
      SUBCASE("valid mapping") {
        unordered_bidict<int, std::string> bd{{1, "one"}, {2, "two"}};
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("invalid mapping") {
        CHECK_THROWS(
            unordered_bidict<int, std::string>{{1, "one"}, {2, "one"}});
      }
    }

    SUBCASE("unordered_bidict::unordered_bidict(InputIt, InputIt)") {
      SUBCASE("valid mapping") {
        std::vector<std::pair<int, std::string>> pairs = {{1, "one"},
                                                          {2, "two"}};
        unordered_bidict<int, std::string> bd{pairs.begin(), pairs.end()};
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("invalid mapping") {
        std::vector<std::pair<int, std::string>> bad_pairs = {{1, "one"},
                                                              {2, "one"}};
        CHECK_THROWS(unordered_bidict<int, std::string>{bad_pairs.begin(),
                                                        bad_pairs.end()});
      }
    }

    SUBCASE("unordered_bidict::unordered_bidict(std::unordered_map<L, R> const "
            "&, std::unordered_map<R, L> const &)") {
      SUBCASE("valid mapping") {
        std::unordered_map<int, std::string> fwd = {{1, "one"}, {2, "two"}};
        std::unordered_map<std::string, int> bwd = {{"one", 1}, {"two", 2}};
        unordered_bidict<int, std::string> bd{fwd, bwd};
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("invalid mapping") {
        std::unordered_map<int, std::string> bad_fwd = {{1, "one"}, {2, "one"}};
        std::unordered_map<std::string, int> bad_bwd = {{"one", 1}};
        CHECK_THROWS(unordered_bidict<int, std::string>{bad_fwd, bad_bwd});
      }
    }

    SUBCASE("unordered_bidict::erase_l") {
      dict.erase_l(1);
      CHECK(dict.size() == 1);
      CHECK_THROWS(dict.at_l(1));
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("unordered_bidict::erase_r") {
      dict.erase_r("one");
      CHECK(dict.size() == 1);
      CHECK_THROWS(dict.at_r("one"));
      CHECK(dict.at_l(2) == "two");
    }

    SUBCASE("unordered_bidict::equate") {
      CHECK_THROWS(dict.equate(1, "three"));
      CHECK_THROWS(dict.equate(3, "two"));

      dict.equate(3, "three");
      CHECK(dict.at_l(3) == "three");
      CHECK(dict.at_r("three") == 3);
    }

    SUBCASE("unordered_bidict::try_equate") {
      SUBCASE("l value already present") {
        bool succeeded = dict.try_equate(1, "three");

        CHECK_FALSE(succeeded);

        unordered_bidict<int, std::string> correct =
            unordered_bidict<int, std::string>{
                {1, "one"},
                {2, "two"},
            };

        CHECK(dict == correct);
      }

      SUBCASE("r value already present") {
        bool succeeded = dict.try_equate(3, "one");

        CHECK_FALSE(succeeded);

        unordered_bidict<int, std::string> correct =
            unordered_bidict<int, std::string>{
                {1, "one"},
                {2, "two"},
            };

        CHECK(dict == correct);
      }

      SUBCASE("both values already present but not bound to each other") {
        bool succeeded = dict.try_equate(1, "two");

        CHECK_FALSE(succeeded);

        unordered_bidict<int, std::string> correct =
            unordered_bidict<int, std::string>{
                {1, "one"},
                {2, "two"},
            };

        CHECK(dict == correct);
      }

      SUBCASE("both valyes already present and bound to each other") {
        bool succeeded = dict.try_equate(1, "one");

        CHECK(succeeded);

        unordered_bidict<int, std::string> correct =
            unordered_bidict<int, std::string>{
                {1, "one"},
                {2, "two"},
            };

        CHECK(dict == correct);
      }

      SUBCASE("neither value already present") {
        bool succeeded = dict.try_equate(3, "three");

        CHECK(succeeded);

        unordered_bidict<int, std::string> correct =
            unordered_bidict<int, std::string>{
                {1, "one"},
                {2, "two"},
                {3, "three"},
            };

        CHECK(dict == correct);
      }
    }

    SUBCASE("unordered_bidict::operator==") {
      SUBCASE("a unordered_bidict is equal to itself") {
        CHECK(dict == dict);
      }

      SUBCASE("a unordered_bidict is equal to a different unordered_bidict "
              "with the same mapping") {
        unordered_bidict<int, std::string> bd{{1, "one"}, {2, "two"}};
        CHECK(dict == bd);
      }

      SUBCASE("nonequal unordered_bidicts are not identified as equal") {
        unordered_bidict<int, std::string> bd2{{1, "one"}, {3, "three"}};
        CHECK_FALSE(dict == bd2);
      }
    }

    SUBCASE("unordered_bidict::operator!=") {
      SUBCASE("it is false that a unordered_bidict is not equal to itself") {
        CHECK_FALSE(dict != dict);
      }

      SUBCASE("it is false that a unordered_bidict is not equal to a different "
              "unordered_bidict "
              "with the same mapping") {
        unordered_bidict<int, std::string> bd{{1, "one"}, {2, "two"}};
        CHECK_FALSE(dict != bd);
      }

      SUBCASE("nonequal unordered_bidicts are identified as not equal") {
        unordered_bidict<int, std::string> bd2{{1, "one"}, {3, "three"}};
        CHECK(dict != bd2);
      }
    }

    SUBCASE("unordered_bidict::at_l") {
      CHECK(dict.at_l(1) == "one");
      CHECK_THROWS(dict.at_l(3));
    }

    SUBCASE("unordered_bidict::at_r") {
      CHECK(dict.at_r("one") == 1);
      CHECK_THROWS(dict.at_r("three"));
    }

    SUBCASE("unordered_bidict::left_values") {
      CHECK(dict.left_values() == std::unordered_set<int>{1, 2});
    }

    SUBCASE("unordered_bidict::right_values") {
      CHECK(dict.right_values() ==
            std::unordered_set<std::string>{"one", "two"});
    }

    SUBCASE("unordered_bidict::size") {
      CHECK(dict.size() == 2);
    }

    SUBCASE("unordered_bidict::empty") {
      SUBCASE("nonempty unordered_bidicts are not identified as empty") {
        CHECK_FALSE(dict.empty());
      }

      SUBCASE("empty unordered_bidicts are identified as empty") {
        unordered_bidict<int, std::string> empty{};
        CHECK(empty.empty());
      }

      SUBCASE("unordered_bidicts emptied by erase are identified as empty") {
        dict.erase_l(1);
        dict.erase_l(2);
        CHECK(dict.empty());
      }
    }

    SUBCASE("unordered_bidict::begin") {
      auto it = dict.begin();
      if (it->first == 1) {
        CHECK(it->second == "one");
      } else {
        CHECK(it->first == 2);
        CHECK(it->second == "two");
      }
    }

    SUBCASE("unordered_bidict::end") {
      auto it = dict.end();

      CHECK_WITHOUT_STRINGIFY(it == dict.end());
    }

    SUBCASE("unordered_bidict::reversed") {
      unordered_bidict<std::string, int> reversed_dict = dict.reversed();
      CHECK(reversed_dict.at_l("one") == 1);
      CHECK(reversed_dict.at_r(2) == "two");

      SUBCASE("a twice-reversed unordered_bidict is equal to itself") {
        unordered_bidict<int, std::string> result = dict.reversed().reversed();

        CHECK(result == dict);
      }
    }

    SUBCASE("implicitly convert to std::unordered_map") {
      std::unordered_map<int, std::string> res = dict;
      std::unordered_map<int, std::string> expected = {{1, "one"}, {2, "two"}};
      CHECK(res == expected);
    }

    SUBCASE("fmt::to_string(unordered_bidict<int, std::string>)") {
      std::string result = fmt::to_string(dict);
      std::string correct = fmt::to_string(dict.as_unordered_map());
      CHECK(result == correct);
    }
  }

  TEST_CASE("adl_serializer<unordered_bidict<L, R>>") {
    unordered_bidict<int, std::string> deserialized =
        unordered_bidict<int, std::string>{
            {2, "hello"},
            {3, "goodbye"},
            {4, "yes"},
        };

    nlohmann::json serialized = std::vector<std::pair<int, std::string>>{
        {2, "hello"},
        {3, "goodbye"},
        {4, "yes"},
    };

    SUBCASE("to_json") {
      nlohmann::json result = deserialized;
      nlohmann::json correct = serialized;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      unordered_bidict<int, std::string> result = serialized;
      unordered_bidict<int, std::string> correct = deserialized;

      CHECK(result == correct);
    }
  }

  TEST_CASE("rc::Arbitrary<unordered_bidict<L, R>>") {
    RC_SUBCASE([](unordered_bidict<int, std::string>) {});
  }
}
