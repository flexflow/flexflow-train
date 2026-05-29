#include "utils/containers/binary_merge_maps_with.h"
#include "test/utils/doctest/fmt/map.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_merge_maps_with") {
    auto fail_if_called = [](std::string const &,
                             std::string const &) -> std::string { PANIC(); };

    SUBCASE("lhs and rhs do not overlap") {
      std::map<int, std::string> lhs = {
          {1, "lhs_one."},
          {4, "lhs_four."},
      };

      std::map<int, std::string> rhs = {
          {2, "rhs_two."},
          {5, "rhs_five."},
      };

      std::map<int, std::string> correct = {
          {1, "lhs_one."},
          {2, "rhs_two."},
          {4, "lhs_four."},
          {5, "rhs_five."},
      };

      std::map<int, std::string> result =
          binary_merge_maps_with(lhs, rhs, fail_if_called);

      CHECK(result == correct);
    }

    SUBCASE("lhs and rhs overlap") {
      std::map<int, std::string> lhs = {
          {1, "lhs_one."},
          {4, "lhs_four."},
      };

      std::map<int, std::string> rhs = {
          {2, "rhs_two."},
          {4, "rhs_four."},
          {5, "rhs_five."},
      };

      std::map<int, std::string> result = binary_merge_maps_with(
          lhs, rhs, [](std::string const &l, std::string const &r) {
            return l + r;
          });

      std::map<int, std::string> correct = {
          {1, "lhs_one."},
          {2, "rhs_two."},
          {4, "lhs_four.rhs_four."},
          {5, "rhs_five."},
      };

      CHECK(result == correct);
    }

    SUBCASE("lhs is empty") {
      std::map<int, std::string> lhs = {};

      std::map<int, std::string> rhs = {
          {2, "rhs_two."},
          {4, "rhs_four."},
          {5, "rhs_five."},
      };

      std::map<int, std::string> result =
          binary_merge_maps_with(lhs, rhs, fail_if_called);

      std::map<int, std::string> correct = rhs;

      CHECK(result == correct);
    }

    SUBCASE("rhs is empty") {
      std::map<int, std::string> lhs = {
          {1, "lhs_one."},
          {4, "lhs_four."},
      };

      std::map<int, std::string> rhs = {};

      std::map<int, std::string> result =
          binary_merge_maps_with(lhs, rhs, fail_if_called);

      std::map<int, std::string> correct = lhs;

      CHECK(result == correct);
    }

    SUBCASE("both lhs and rhs are empty") {
      std::map<int, std::string> lhs = {};

      std::map<int, std::string> rhs = {};

      std::map<int, std::string> result =
          binary_merge_maps_with(lhs, rhs, fail_if_called);

      std::map<int, std::string> correct = {};

      CHECK(result == correct);
    }
  }
}
