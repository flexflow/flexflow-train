#include "utils/one_to_many/one_to_many.h"
#include "utils/one_to_many/one_to_many_from_l_to_r_mapping.h"
#include <doctest/doctest.h>
#include "utils/containers/multiset_of.h"
#include "test/utils/doctest/fmt/multiset.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("OneToMany") {
    FAIL("TODO");
  }

  TEST_CASE("fmt::to_string(OneToMany<L, R>)") {
    OneToMany<int, std::string> input 
      = one_to_many_from_l_to_r_mapping<int, std::string>({
        {1, {"hello", "world"}},
        {2, {}},
        {3, {"HELLO"}}
      });

    std::string result = fmt::to_string(input);
    std::string correct = "{{1, {hello, world}}, {3, {HELLO}}}";

    CHECK(multiset_of(result) == multiset_of(correct));
  }
}
