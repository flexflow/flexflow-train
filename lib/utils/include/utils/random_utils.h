#ifndef _RANDOM_UTILS_H
#define _RANDOM_UTILS_H

#include "utils/containers/all_of.h"
#include <cstdlib>
#include <libassert/assert.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace FlexFlow {

template <typename Generator>
float randf(Generator &g) {
  std::uniform_real_distribution<> dist(0.0, 1.0);
  return dist(g);
}

template <typename Generator, typename T>
T select_random(Generator &g, std::vector<T> const &values) {
  ASSERT(!values.empty());

  std::uniform_int_distribution<> dist(0, values.size() - 1);
  return values[dist(g)];
}

template <typename Generator, typename T>
T select_random(Generator &g,
                std::vector<T> const &values,
                std::vector<float> const &weights) {
  ASSERT(values.size() == weights.size());
  ASSERT(all_of(weights, [](float w) -> bool { return w >= 0; }));

  std::discrete_distribution<> dist(weights.cbegin(), weights.cend());
  return values.at(dist(g));
}

} // namespace FlexFlow

#endif // _RANDOM_UTILS_H
