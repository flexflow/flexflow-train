#include "utils/positive_int/positive_int.h"
#include <doctest/doctest.h>
#include "test/utils/rapidcheck.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("positive_int{int}") {
    int x1 = 3;
    int x2 = 4;

    int zero = 0;
    int negative = -3;

    CHECK(positive_int{x1} == positive_int{x1});
    CHECK(positive_int{x2} != positive_int{x1});

    CHECK_THROWS(positive_int{zero});
    CHECK_THROWS(positive_int{negative});
  }

  TEST_CASE("positive_int{size_t}") {
    size_t x1 = 3;
    size_t x2 = 4;

    size_t zero = 0;

    size_t maxint = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t overflow1 = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    size_t overflow2 = static_cast<size_t>(std::numeric_limits<int>::max()) + 2;

    CHECK(positive_int{x1} == positive_int{x1});
    CHECK(positive_int{x2} != positive_int{x1});

    CHECK_THROWS(positive_int{zero});
    CHECK(positive_int{maxint} == positive_int{maxint});
    CHECK_THROWS(positive_int{overflow1});
    CHECK_THROWS(positive_int{overflow2});
  }

  TEST_CASE("positive_int{nonnegative_int}") {
    nonnegative_int x1 = 3_n;
    nonnegative_int x2 = 4_n;

    nonnegative_int zero = 0_n;

    CHECK(positive_int{x1} == positive_int{x1});
    CHECK(positive_int{x2} != positive_int{x1});

    CHECK_THROWS(positive_int{zero});
  }

  TEST_CASE("_p notation for positive_int") {
    CHECK(9_p == positive_int{9}); 
    CHECK_THROWS(0_p);
  }

  TEST_CASE("static_cast<int>(positive_int)") {
    CHECK(static_cast<int>(8_p) == 8);   
  }

  TEST_CASE("static_cast<nonnegative_int>(positive_int)") {
    CHECK(static_cast<nonnegative_int>(6_p) == 6);   
  }

  TEST_CASE("positive_int < positive_int") {
    CHECK(4_p < 5_p);
    CHECK_FALSE(7_p < 7_p);
    CHECK_FALSE(3_p < 2_p);
  }

  TEST_CASE("positive_int == positive_int") {
    CHECK_FALSE(4_p == 5_p);
    CHECK(7_p == 7_p);
    CHECK_FALSE(3_p == 2_p);
  }

  TEST_CASE("positive_int > positive_int") {
    CHECK_FALSE(4_p > 5_p);
    CHECK_FALSE(7_p > 7_p);
    CHECK(3_p > 2_p);
  }

  TEST_CASE("positive_int <= positive_int") {
    CHECK(4_p <= 5_p);
    CHECK(7_p <= 7_p);
    CHECK_FALSE(3_p <= 2_p);
  }

  TEST_CASE("positive_int != positive_int") {
    CHECK(4_p != 5_p);
    CHECK_FALSE(7_p != 7_p);
    CHECK(3_p != 2_p);
  }

  TEST_CASE("positive_int >= positive_int") {
    CHECK_FALSE(4_p >= 5_p);
    CHECK(7_p >= 7_p);
    CHECK(3_p >= 2_p);
  }

  TEST_CASE("positive_int < nonnegative_int") {
    CHECK(4_p < 5_n);
    CHECK_FALSE(7_p < 7_n);
    CHECK_FALSE(3_p < 2_n);
    CHECK_FALSE(1_p < 0_n);
  }

  TEST_CASE("positive_int == nonnegative_int") {
    CHECK_FALSE(4_p == 5_n);
    CHECK(7_p == 7_n);
    CHECK_FALSE(3_p == 2_n);
    CHECK_FALSE(1_p == 0_n);
  }

  TEST_CASE("positive_int > nonnegative_int") {
    CHECK_FALSE(4_p > 5_n);
    CHECK_FALSE(7_p > 7_n);
    CHECK(3_p > 2_n);
    CHECK(1_p > 0_n);
  }

  TEST_CASE("positive_int <= nonnegative_int") {
    CHECK(4_p <= 5_n);
    CHECK(7_p <= 7_n);
    CHECK_FALSE(3_p <= 2_n);
    CHECK_FALSE(1_p <= 0_n);
  }

  TEST_CASE("positive_int != nonnegative_int") {
    CHECK(4_p != 5_n);
    CHECK_FALSE(7_p != 7_n);
    CHECK(3_p != 2_n);
    CHECK(1_p != 0_n);
  }

  TEST_CASE("positive_int >= nonnegative_int") {
    CHECK_FALSE(4_p >= 5_n);
    CHECK(7_p >= 7_n);
    CHECK(3_p >= 2_n);
    CHECK(1_p >= 0_n);
  }

  TEST_CASE("nonnegative_int < positive_int") {
    CHECK(4_n < 5_p);
    CHECK_FALSE(7_n < 7_p);
    CHECK_FALSE(3_n < 2_p);
    CHECK(0_n < 1_p);
  }

  TEST_CASE("nonnegative_int == positive_int") {
    CHECK_FALSE(4_n == 5_p);
    CHECK(7_n == 7_p);
    CHECK_FALSE(3_n == 2_p);
    CHECK_FALSE(0_n == 1_p);
  }

  TEST_CASE("nonnegative_int > positive_int") {
    CHECK_FALSE(4_n > 5_p);
    CHECK_FALSE(7_n > 7_p);
    CHECK(3_n > 2_p);
    CHECK_FALSE(0_n > 1_p);
  }

  TEST_CASE("nonnegative_int <= positive_int") {
    CHECK(4_n <= 5_p);
    CHECK(7_n <= 7_p);
    CHECK_FALSE(3_n <= 2_p);
    CHECK(0_n <= 1_p);
  }

  TEST_CASE("nonnegative_int != positive_int") {
    CHECK(4_n != 5_p);
    CHECK_FALSE(7_n != 7_p);
    CHECK(3_n != 2_p);
    CHECK(0_n != 1_p);
  }

  TEST_CASE("nonnegative_int >= positive_int") {
    CHECK_FALSE(4_n >= 5_p);
    CHECK(7_n >= 7_p);
    CHECK(3_n >= 2_p);
    CHECK_FALSE(0_n >= 1_p);
  }

  TEST_CASE("positive_int < int") {
    CHECK(4_p < 5);
    CHECK_FALSE(7_p < 7);
    CHECK_FALSE(3_p < 2);
    CHECK_FALSE(1_p < -3);
  }

  TEST_CASE("positive_int == int") {
    CHECK_FALSE(4_p == 5);
    CHECK(7_p == 7);
    CHECK_FALSE(3_p == 2);
    CHECK_FALSE(1_p == -3);
  }

  TEST_CASE("positive_int > int") {
    CHECK_FALSE(4_p > 5);
    CHECK_FALSE(7_p > 7);
    CHECK(3_p > 2);
    CHECK(1_p > -3);
  }

  TEST_CASE("positive_int <= int") {
    CHECK(4_p <= 5);
    CHECK(7_p <= 7);
    CHECK_FALSE(3_p <= 2);
    CHECK_FALSE(1_p <= -3);
  }

  TEST_CASE("positive_int != int") {
    CHECK(4_p != 5);
    CHECK_FALSE(7_p != 7);
    CHECK(3_p != 2);
    CHECK(1_p != -3);
  }

  TEST_CASE("positive_int >= int") {
    CHECK_FALSE(4_p >= 5);
    CHECK(7_p >= 7);
    CHECK(3_p >= 2);
    CHECK(1_p >= -3);
  }

  TEST_CASE("int < positive_int") {
    CHECK(4 < 5_p);
    CHECK_FALSE(7 < 7_p);
    CHECK_FALSE(3 < 2_p);
    CHECK(-3 < 1_p);
  }

  TEST_CASE("int == positive_int") {
    CHECK_FALSE(4 == 5_p);
    CHECK(7 == 7_p);
    CHECK_FALSE(3 == 2_p);
    CHECK_FALSE(-3 == 1_p);
  }

  TEST_CASE("int > positive_int") {
    CHECK_FALSE(4 > 5_p);
    CHECK_FALSE(7 > 7_p);
    CHECK(3 > 2_p);
    CHECK_FALSE(-3 > 1_p);
  }

  TEST_CASE("int <= positive_int") {
    CHECK(4 <= 5_p);
    CHECK(7 <= 7_p);
    CHECK_FALSE(3 <= 2_p);
    CHECK(-3 <= 1_p);
  }

  TEST_CASE("int != positive_int") {
    CHECK(4 != 5_p);
    CHECK_FALSE(7 != 7_p);
    CHECK(3 != 2_p);
    CHECK(-3 != 1_p);
  }

  TEST_CASE("int >= positive_int") {
    CHECK_FALSE(4 >= 5_p);
    CHECK(7 >= 7_p);
    CHECK(3 >= 2_p);
    CHECK_FALSE(-3 >= 1_p);
  }

  TEST_CASE("positive_int + positive_int") {
    CHECK(4_p + 2_p == 6_p);
  }

  TEST_CASE("positive_int + nonnegative_int") {
    CHECK(4_p + 3_n == 7_p);
  }

  TEST_CASE("++positive_int") {
    positive_int x = 3_p;
    CHECK(++x == 4_p);
    CHECK(x == 4_p);
  }

  TEST_CASE("positive_int++") {
    positive_int x = 3_p;
    CHECK(x++ == 3_p);
    CHECK(x == 4_p);
  }

  TEST_CASE("positive_int += positive_int ") {
    positive_int x = 3_p;

    SUBCASE("single application") {
      CHECK((x += 2_p) == 5_p);
      CHECK(x == 5_p);
    }

    SUBCASE("repeated application") {
      CHECK(((x += 2_p) += 4_p) == 9_p);
      CHECK(x == 9_p);
    }
  }

  TEST_CASE("positive_int += nonnegative_int") {
    positive_int x = 3_p;

    SUBCASE("rhs is positive") {
      CHECK((x += 2_n) == 5_p);
      CHECK(x == 5_p);
    }

    SUBCASE("rhs is zero") {
      CHECK((x += 0_n) == 3_p);
      CHECK(x == 3_p);
    }

    SUBCASE("repeated application") {
      CHECK(((x += 2_n) += 4_n) == 9_p);
      CHECK(x == 9_p);
    }
  }

  TEST_CASE("positive_int * positive_int") {
    CHECK(3_p * 4_p == 12_p); 
  }

  TEST_CASE("positive_int *= positive_int") {
    positive_int x = 5_p;

    SUBCASE("single application") {
      CHECK((x *= 2_p) == 10_p);
      CHECK(x == 10_p);
    }

    SUBCASE("repeated application") {
      CHECK(((x *= 2_p) *= 3_p) == 30_p);
      CHECK(x == 30_p);
    }
  }

  TEST_CASE("positive_int * nonnegative_int") {
    CHECK(3_p * 4_n == 12_n); 
    CHECK(3_p * 0_n == 0_n);
  }

  TEST_CASE("positive_int / positive_int") {
    CHECK(4_p / 2_p == 2_n);
    CHECK(4_p / 3_p == 1_n);
    CHECK(4_p / 4_p == 1_n);
    CHECK(4_p / 5_p == 0_n);
  }

  TEST_CASE("nonnegative_int / positive_int") {
    CHECK(4_n / 2_p == 2_n);
    CHECK(4_n / 3_p == 1_n);
    CHECK(4_n / 4_p == 1_n);
    CHECK(4_n / 5_p == 0_n);

    CHECK(0_n / 1_p == 0_n);
  }

  TEST_CASE("float / positive_int") {
    CHECK(4.0f / 2_p == 2.0f);   
    CHECK(3.0f / 2_p == 1.5f);   
    CHECK(-3.0f / 4_p == -0.75f);   
    CHECK(0.0f / 1_p == 0.0f);   
  }

  TEST_CASE("float /= positive_int") {
    SUBCASE("divides evenly") {
      float x = 4.0f;
      CHECK((x /= 2_p) == 2.0f);
      CHECK(x == 2.0f);
    }

    SUBCASE("does not divide evenly") {
      float x = 3.0f;
      CHECK((x /= 2_p) == 1.5f);
      CHECK(x == 1.5f);
    }

    SUBCASE("numerator is negative") {
      float x = -3.0f;
      CHECK((x /= 4_p) == -0.75f);
      CHECK(x == -0.75f);
    }

    SUBCASE("numerator is zero") {
      float x = 0.0f;
      CHECK((x /= 4_p) == 0.0f);
      CHECK(x == 0.0f);
    }

    SUBCASE("repeated /=") {
      float x = 20.0f;
      int x2 = 20;
      CHECK(((x /= 4_p) /= 2_p) == 2.5f);
      CHECK(x == 2.5f);
    }
  }

  TEST_CASE("positive_int % positive_int") {
    CHECK(4_p % 3_p == 1_n);
    CHECK(5_p % 5_p == 0_n);
  }

  TEST_CASE("nonnegative_int % positive_int") {
    CHECK(4_n % 3_p == 1_n);
    CHECK(5_n % 5_p == 0_n);
    CHECK(0_n % 3_p == 0_n);
  }

  TEST_CASE("positive_int::int_from_positive_int()") {
    CHECK((3_p).int_from_positive_int() == 3); 
  }

  TEST_CASE("positive_int::nonnegative_int_from_positive_int()") {
    CHECK((4_p).nonnegative_int_from_positive_int() == 4); 
  }

  TEST_CASE("positive_int::operator<<(std::ostream &, positive_int)") {
    std::ostringstream oss;
    oss << 3_p;

    std::string result = oss.str();
    std::string correct = "3";

    CHECK(result == correct);
  }

  TEST_CASE("positive_int fmt support") {
    std::string result = fmt::to_string(14_p);
    std::string correct = "14";

    CHECK(result == correct);
  }

  TEST_CASE("adl_serializer<positive_int>") {
    SUBCASE("to_json") {
      positive_int input = 5_p;

      nlohmann::json result = input;
      nlohmann::json correct = 5;

      CHECK(result == correct);
    } 

    SUBCASE("from_json") {
      nlohmann::json input = 5;

      positive_int result = input.template get<positive_int>();
      positive_int correct = 5_p;

      CHECK(result == correct);
    }
  }

  TEST_CASE("std::hash<nonnegative_int>") {
    positive_int nn_int_1a = positive_int{1};
    positive_int nn_int_1b = positive_int{1};
    positive_int nn_int_2 = positive_int{2};
    std::hash<positive_int> hash_fn;

    SUBCASE("Identical values have the same hash") {
      CHECK(hash_fn(nn_int_1a) == hash_fn(nn_int_1b));
    }

    SUBCASE("Different values have different hashes") {
      CHECK(hash_fn(nn_int_1a) != hash_fn(nn_int_2));
    }

    SUBCASE("unordered_set works with positive_int") {
      std::unordered_set<::FlexFlow::positive_int> positive_int_set;
      positive_int_set.insert(nn_int_1a);
      positive_int_set.insert(nn_int_1b);
      positive_int_set.insert(nn_int_2);

      CHECK(positive_int_set.size() == 2);
    }
  }

  TEST_CASE("rc::Arbitrary<positive_int>") {
    RC_SUBCASE([](positive_int) { });
  }
}
