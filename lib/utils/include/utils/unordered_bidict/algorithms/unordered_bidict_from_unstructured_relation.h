#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_UNSTRUCTURED_RELATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_UNSTRUCTURED_RELATION_H

#include "utils/unordered_bidict/unordered_bidict.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/get_element_counts.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename L, typename R>
unordered_bidict<L, R> unordered_bidict_from_unstructured_relation(
    std::unordered_set<std::pair<L, R>> const &relation) {

  {
    std::unordered_multiset<L> l_values =
        transform(unordered_multiset_of(relation),
                  [](std::pair<L, R> const &p) -> L { return p.first; });

    std::unordered_map<L, positive_int> l_value_counts = get_element_counts(l_values);

    std::unordered_map<L, positive_int> duplicated_element_counts =
        filter_values(l_value_counts, [](positive_int num_occurences) -> bool {
          return num_occurences > 1;
        });

    ASSERT(duplicated_element_counts.empty(), duplicated_element_counts);
  }

  {
    std::unordered_multiset<R> r_values =
        transform(unordered_multiset_of(relation),
                  [](std::pair<L, R> const &p) -> R { return p.second; });

    std::unordered_map<R, positive_int> r_value_counts = get_element_counts(r_values);

    std::unordered_map<R, positive_int> duplicated_element_counts =
        filter_values(r_value_counts, [](positive_int num_occurences) -> bool {
          return num_occurences > 1;
        });

    ASSERT(duplicated_element_counts.empty(), duplicated_element_counts);
  }

  return unordered_bidict<L, R>{relation.begin(), relation.end()};
}

} // namespace FlexFlow

#endif
