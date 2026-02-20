#include "utils/fmt/expected.h"
#include <doctest/doctest.h>
#include <fmt/format.h>
#include <sstream>
#include <tl/expected.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

namespace doctest {

#define CHECK_WITHOUT_STRINGIFY(...)                                           \
  do {                                                                         \
    bool result = __VA_ARGS__;                                                 \
    CHECK(result);                                                             \
  } while (0);

#define CHECK_FALSE_WITHOUT_STRINGIFY(...)                                     \
  do {                                                                         \
    bool result = __VA_ARGS__;                                                 \
    CHECK_FALSE(result);                                                       \
  } while (0);

} // namespace doctest
