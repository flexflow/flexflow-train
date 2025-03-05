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
      CHECK(nn_int_1a == nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, not equal") {
      CHECK_FALSE(nn_int_1a == nn_int_2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equal") {
      CHECK(nn_int_1a == 1);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, not equal") {
      CHECK_FALSE(nn_int_1a == 2);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equal") {
      CHECK(1 == nn_int_1b);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, not equal") {
      CHECK_FALSE(2 == nn_int_1b);
    }
  }

  TEST_CASE("nonnegative_int != comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equal") {
      CHECK_FALSE(nn_int_1a != nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, not equal") {
      CHECK(nn_int_1a != nn_int_2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equal") {
      CHECK_FALSE(nn_int_1a != 1);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, not equal") {
      CHECK(nn_int_1a != 2);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equal") {
      CHECK_FALSE(1 != nn_int_1b);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, not equal") {
      CHECK(2 != nn_int_1b);
    }
  }

  TEST_CASE("nonnegative_int < comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK(nn_int_1a < nn_int_2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK_FALSE(nn_int_1a < nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK_FALSE(nn_int_2 < nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK(nn_int_1a < 2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK_FALSE(nn_int_1a < 1);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK_FALSE(nn_int_2 < 1);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK(1 < nn_int_2);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK_FALSE(1 < nn_int_1b);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK_FALSE(2 < nn_int_1b);
    }
  }

  TEST_CASE("nonnegative_int <= comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK(nn_int_1a <= nn_int_2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK(nn_int_1a <= nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK_FALSE(nn_int_2 <= nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK(nn_int_1a <= 2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK(nn_int_1a <= 1);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK_FALSE(nn_int_2 <= 1);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK(1 <= nn_int_2);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK(1 <= nn_int_1b);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK_FALSE(2 <= nn_int_1b);
    }
  }

  TEST_CASE("nonnegative_int > comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK_FALSE(nn_int_1a > nn_int_2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK_FALSE(nn_int_1a > nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK(nn_int_2 > nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK_FALSE(nn_int_1a > 2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK_FALSE(nn_int_1a > 1);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK(nn_int_2 > 1);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK_FALSE(1 > nn_int_2);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK_FALSE(1 > nn_int_1b);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK(2 > nn_int_1b);
    }
  }

  TEST_CASE("nonnegative_int >= comparisons") {
    nonnegative_int nn_int_1a = nonnegative_int{1};
    nonnegative_int nn_int_1b = nonnegative_int{1};
    nonnegative_int nn_int_2 = nonnegative_int{2};
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, less than") {
      CHECK_FALSE(nn_int_1a >= nn_int_2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, equals") {
      CHECK(nn_int_1a >= nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: nonnegative_int, greater than") {
      CHECK(nn_int_2 >= nn_int_1b);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, less than") {
      CHECK_FALSE(nn_int_1a >= 2);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, equals") {
      CHECK(nn_int_1a >= 1);
    }
    SUBCASE("LHS: nonnegative_int, RHS: int, greater than") {
      CHECK(nn_int_2 >= 1);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, less than") {
      CHECK_FALSE(1 >= nn_int_2);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, equals") {
      CHECK(1 >= nn_int_1b);
    }
    SUBCASE("LHS: int, RHS: nonnegative_int, greater than") {
      CHECK(2 >= nn_int_1b);
    }
  }

  TEST_CASE("nonnegative_int::operator+(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{1} + nonnegative_int{2};
    nonnegative_int correct = nonnegative_int{3};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator++() (pre-increment)") {
    nonnegative_int input = nonnegative_int{1};

    nonnegative_int result = ++input;
    nonnegative_int correct = nonnegative_int{2};

    CHECK(result == correct);
    CHECK(input == correct);
  }

  TEST_CASE("nonnegative_int::operator++(int) (post-increment)") {
    nonnegative_int input = nonnegative_int{1};

    nonnegative_int result = input++;
    nonnegative_int correct_input = nonnegative_int{2};
    nonnegative_int correct_result = nonnegative_int{1};

    CHECK(result == correct_result);
    CHECK(input == correct_input);
  }

  TEST_CASE("nonnegative_int::operator+=(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{1};
    result += nonnegative_int{3};

    nonnegative_int correct = nonnegative_int{4};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator*(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{2} * nonnegative_int{3};
    nonnegative_int correct = nonnegative_int{6};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator*=(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{3};
    result *= nonnegative_int{6};

    nonnegative_int correct = nonnegative_int{18};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator/(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{5} / nonnegative_int{2};
    nonnegative_int correct = nonnegative_int{2};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator/=(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{13};
    result /= nonnegative_int{3};

    nonnegative_int correct = nonnegative_int{4};

    CHECK(result == correct);
  }

  TEST_CASE("operator/(float, nonnegative_int)") {
    float result = 5.0 / nonnegative_int{2};
    float correct = 5.0 / 2;

    CHECK(result == correct);
  }

  TEST_CASE("operator/=(float, nonnegative_int)") {
    float result = 13.0;
    result /= nonnegative_int{3};

    float correct = 13.0 / 3;

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator/(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{5} / nonnegative_int{2};
    nonnegative_int correct = nonnegative_int{2};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator/=(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{13};
    result /= nonnegative_int{3};

    nonnegative_int correct = nonnegative_int{4};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator%(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{5} % nonnegative_int{2};
    nonnegative_int correct = nonnegative_int{1};

    CHECK(result == correct);
  }

  TEST_CASE("nonnegative_int::operator%=(nonnegative_int)") {
    nonnegative_int result = nonnegative_int{15};
    result %= nonnegative_int{4};

    nonnegative_int correct = nonnegative_int{3};

    CHECK(result == correct);
  }

  TEST_CASE("adl_serializer<nonnegative_int>") {
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
      std::unordered_set<::FlexFlow::nonnegative_int> nonnegative_int_set;
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

  TEST_CASE("fmt::to_string(nonnegative_int)") {
    nonnegative_int nn_int_1 = nonnegative_int{1};
    CHECK(fmt::to_string(nn_int_1) == "1");
  }
}
