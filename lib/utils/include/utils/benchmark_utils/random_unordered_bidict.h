#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_UNORDERED_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_UNORDERED_BIDICT_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/unordered_bidict/unordered_bidict.h"
#include <libassert/assert.hpp>
#include <random>

namespace FlexFlow {

template <typename FL,
          typename FR,
          typename L = std::invoke_result_t<FL, std::mt19937 &>,
          typename R = std::invoke_result_t<FR, std::mt19937 &>>
unordered_bidict<L, R> random_unordered_bidict(std::mt19937 &gen,
                                               nonnegative_int size,
                                               FL &&fl,
                                               FR &&fr) {
  unordered_bidict<L, R> result;

  auto generate_l = [&]() -> L {
    while (true) {
      L l = fl(gen);
      if (!result.contains_l(l)) {
        return l;
      }
    }
  };

  auto generate_r = [&]() -> R {
    while (true) {
      R r = fr(gen);
      if (!result.contains_r(r)) {
        return r;
      }
    }
  };

  while (result.size() < size) {
    L l = generate_l();
    R r = generate_r();
    result.equate(l, r);
  }

  DEBUG_ASSERT(result.size() == size);

  return result;
}

} // namespace FlexFlow

#endif
