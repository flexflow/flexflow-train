#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("nonnegative_int initialization") {
    SUBCASE("positive int initialization") {
      CHECK_NOTHROW(nonnegative_int{1});
    }

    SUBCASE("zero initialization") {
      CHECK_NOTHROW(nonnegative_int{0});
    }

    SUBCASE("negative int initialization") {
      CHECK_THROWS(nonnegative_int{-1});
    }
  }

  TEST_CASE("nonnegative_int == comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equal") {
      CHECK((nn_int_1a == nn_int_1b) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, not equal") {
      CHECK((nn_int_1a == nn_int_2) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equal") {
      CHECK((nn_int_1a == 1) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, not equal") {
      CHECK((nn_int_1a == 2) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equal") {
      CHECK((1 == nn_int_1b) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, not equal") {
      CHECK((2 == nn_int_1b) == false);
    }
  }

  TEST_CASE("nonnegative_int != comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equal") {
      CHECK((nn_int_1a != nn_int_1b) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, not equal") {
      CHECK((nn_int_1a != nn_int_2) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equal") {
      CHECK((nn_int_1a != 1) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, not equal") {
      CHECK((nn_int_1a != 2) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equal") {
      CHECK((1 != nn_int_1b) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, not equal") {
      CHECK((2 != nn_int_1b) == true);
    }
  }

  TEST_CASE("nonnegative_int < comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK((nn_int_1a < nn_int_2) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK((nn_int_1a < nn_int_1b) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK((nn_int_2 < nn_int_1b) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK((nn_int_1a < 2) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK((nn_int_1a < 1) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK((nn_int_2 < 1) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK((1 < nn_int_2) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK((1 < nn_int_1b) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK((2 < nn_int_1b) == false);
    }
  }

  TEST_CASE("nonnegative_int <= comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK((nn_int_1a <= nn_int_2) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK((nn_int_1a <= nn_int_1b) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK((nn_int_2 <= nn_int_1b) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK((nn_int_1a <= 2) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK((nn_int_1a <= 1) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK((nn_int_2 <= 1) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK((1 <= nn_int_2) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK((1 <= nn_int_1b) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK((2 <= nn_int_1b) == false);
    }
  }

  TEST_CASE("nonnegative_int > comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK((nn_int_1a > nn_int_2) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK((nn_int_1a > nn_int_1b) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK((nn_int_2 > nn_int_1b) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK((nn_int_1a > 2) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK((nn_int_1a > 1) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK((nn_int_2 > 1) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK((1 > nn_int_2) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK((1 > nn_int_1b) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK((2 > nn_int_1b) == true);
    }
  }

  TEST_CASE("nonnegative_int >= comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK((nn_int_1a >= nn_int_2) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK((nn_int_1a >= nn_int_1b) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK((nn_int_2 >= nn_int_1b) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK((nn_int_1a >= 2) == false);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK((nn_int_1a >= 1) == true);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK((nn_int_2 >= 1) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK((1 >= nn_int_2) == false);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK((1 >= nn_int_1b) == true);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK((2 >= nn_int_1b) == true);
    }
  }

  TEST_CASE("nonnegative_int adl_serializer") {
    SUBCASE("to_json") {
      nonnegative_int input = nonnegative_int{5};

      nlohmann::json result = input;
      nlohmann::json correct = 5;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      nlohmann::json input = 5;

      nonnegative_int result = input.template get<nonnegative_int>();
      nonnegative_int correct = nonnegative_int{5};

      CHECK(result == correct);
    }
  }

  TEST_CASE("std::hash<nonnegative_int>") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    std::hash<nonnegative_int> hash_fn;
    SUBCASE("Identical values have the same hash") {
      CHECK(hash_fn(nn_int_1a) == hash_fn(nn_int_1b));
    }
    SUBCASE("Different values have different hashes") {
      CHECK(hash_fn(nn_int_1a) != hash_fn(nn_int_2));
    }
    SUBCASE("Unordered set works with nonnegative_int") {
      std::unordered_set<FlexFlow::nonnegative_int> nonnegative_int_set;
      nonnegative_int_set.insert(nn_int_1a);
      nonnegative_int_set.insert(nn_int_1b);
      nonnegative_int_set.insert(nn_int_2);

      CHECK(nonnegative_int_set.size() == 2);
    }
  }

  TEST_CASE("nonnegative int >> operator") {
    nonnegative_int nn_int_1 = nonnegative_int{1};
    std::ostringstream oss;
    oss << nn_int_1;

    CHECK(oss.str() == "1");
  }
}
