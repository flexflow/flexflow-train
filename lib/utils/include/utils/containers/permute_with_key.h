#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PERMUTE_WITH_KEY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PERMUTE_WITH_KEY_H

#include "utils/containers/transform.h"
#include "utils/containers/range.h"
#include "utils/containers/product.h"

namespace FlexFlow {

template <typename T>
std::vector<T> permute_with_key(int key, std::vector<T> const &v) {
  int max_permutations = 10000;
  int reduced_key = key % max_permutations;

  std::vector<int> permutation = range(v.size());

  for (int i = 0; i < reduced_key; i++) {
    std::next_permutation(permutation.begin(), permutation.end());
  }

  return transform(permutation, 
                   [&](int permutation_entry) {
                     return v.at(permutation_entry);
                   });
};

} // namespace FlexFlow

#endif
