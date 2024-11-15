#include "utils/stack_vector/stack_vector_of.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("stack_vector_of(std::vector<T>)") {
    std::vector<int> input = {1, 2, 3};
    const size_t MAXSIZE = 5;
    stack_vector<int, MAXSIZE> result = stack_vector_of<MAXSIZE>(input);
    stack_vector<int, MAXSIZE> correct = {1, 2, 3};

    CHECK(result == correct);
  }
}
