#include "utils/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/hash-utils.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

struct Unsortable {
public:
  bool operator==(Unsortable const &other) const {
    return this->value == other.value;
  }

  bool operator!=(Unsortable const &other) const {
    return this->value != other.value;
  }

public:
  int value;
};

std::string format_as(Unsortable const &u) {
  return fmt::to_string(u.value);
}

namespace std {

template <>
struct hash<Unsortable> {
  size_t operator()(Unsortable const &u) const {
    return get_std_hash(u.value);
  };
};

} // namespace std

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::unordered_set<int>)") {
    std::unordered_set<int> input = {0, 1, 3, 2};
    std::string result = fmt::to_string(input);
    std::string correct = "{0, 1, 2, 3}";
    CHECK(result == correct);
  }

  TEST_CASE("fmt::to_string(std::unordered_set<Unsortable>)") {
    std::unordered_set<Unsortable> input = {
        Unsortable{0}, Unsortable{1}, Unsortable{3}, Unsortable{2}};
    std::string result = fmt::to_string(input);
    std::string correct = "{0, 1, 2, 3}";
    CHECK(result != correct);
    CHECK(unordered_multiset_of(result) == unordered_multiset_of(correct));
  }
}
