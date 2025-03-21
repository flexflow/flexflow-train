#include "utils/stack_vector/stack_vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include "utils/archetypes/value_type.h"
#include <doctest/doctest.h>
#include <iterator>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("operator<(stack_vector<T, MAXSIZE>, stack_vector<T, MAXSIZE>)") {
    constexpr std::size_t MAXSIZE = 5;

    SUBCASE("T is ordered") {
      SUBCASE("inputs are the same") {
        std::vector<int> input = {2, 1, 2, 3};

        bool result = (input < input);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("lhs is strict prefix of rhs") {
        std::vector<int> lhs = {2, 1, 2};
        std::vector<int> rhs = {2, 1, 2, 3};

        bool result = (lhs < rhs);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("lhs is empty") {
        std::vector<int> lhs = {};
        std::vector<int> rhs = {2, 1, 2, 3};

        bool result = (lhs < rhs);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("lhs has a smaller element first") {
        std::vector<int> lhs = {2, 1, 0, 3};
        std::vector<int> rhs = {2, 1, 2};

        bool result = (lhs < rhs);
        bool correct = true;

        CHECK(result == correct);
      }

      // from the definition of a strict total order, i.e.,
      // https://en.wikipedia.org/w/index.php?title=Total_order&oldid=1278541072#Strict_and_non-strict_total_orders
      RC_SUBCASE("operator< is irreflexive",
                 [](stack_vector<int, MAXSIZE> const &input) {
                   RC_ASSERT(!(input < input));
                 });

      RC_SUBCASE("operator< is asymmetric",
                 [](stack_vector<int, MAXSIZE> const &lhs,
                    stack_vector<int, MAXSIZE> const &rhs) {
                   RC_PRE(lhs != rhs);

                   RC_ASSERT((lhs < rhs) == !(rhs < lhs));
                 });

      RC_SUBCASE("operator< is transitive",
                 [](stack_vector<int, MAXSIZE> const &a,
                    stack_vector<int, MAXSIZE> const &b,
                    stack_vector<int, MAXSIZE> const &c) {
                   RC_PRE(a < b);
                   RC_PRE(b < c);

                   RC_ASSERT(a < c);
                 });

      RC_SUBCASE("operator< is connected",
                 [](stack_vector<int, MAXSIZE> const &lhs,
                    stack_vector<int, MAXSIZE> const &rhs) {
                   RC_PRE(lhs != rhs);

                   RC_ASSERT((lhs < rhs) || (rhs < lhs));
                 });
    }

    SUBCASE("T is not ordered") {
      bool result = is_lt_comparable_v<stack_vector<value_type<0>, MAXSIZE>>;

      CHECK_FALSE(result);
    }
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE>::push_back", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    std::vector<T> result = vector;
    std::vector<T> correct = {10};
    CHECK(result == correct);

    vector.push_back(20);
    correct = {10, 20};
    result = vector;
    CHECK(result == correct);
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE>::operator[]", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    vector.push_back(20);
    vector.push_back(30);

    CHECK(vector[0] == 10);
    CHECK(vector[1] == 20);
    CHECK(vector[2] == 30);
  }

  TEST_CASE_TEMPLATE("stack_vector<T, MAXSIZE>::size", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    CHECK(vector.size() == 0);

    vector.push_back(10);
    CHECK(vector.size() == 1);

    vector.push_back(20);
    CHECK(vector.size() == 2);
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE>::operator==", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector1, vector2;

    vector1.push_back(10);
    vector1.push_back(15);
    vector1.push_back(20);

    vector2.push_back(10);
    vector2.push_back(15);
    vector2.push_back(20);

    CHECK(vector1 == vector2);
  }

  TEST_CASE_TEMPLATE("stack_vector<T, MAXSIZE>::back", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    CHECK(vector.back() == 10);

    vector.push_back(20);
    CHECK(vector.back() == 20);
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE> - check for size bound", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 10;
    RC_SUBCASE("within bound", [&](stack_vector<T, MAXSIZE> v) {
      RC_ASSERT(v.size() <= MAXSIZE);
    });
  }
}
