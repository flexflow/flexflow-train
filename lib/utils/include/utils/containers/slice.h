#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUBVEC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUBVEC_H

#include "utils/exception.h"
#include <optional>
#include <stdexcept>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> slice(std::vector<T> const &v,
                     int const &maybe_start,
                     std::optional<int> const &maybe_end) {
  auto begin_iter = v.cbegin();
  auto end_iter = v.cend();

  auto resolve_loc = [&](int idx) ->
      typename std::vector<T>::iterator::difference_type {
        int size = static_cast<int>(v.size());
        int new_idx = idx;
        if (idx < 0) {
          new_idx = size + idx;
        }

        ASSERT(new_idx >= 0, "Index out of bounds");
        ASSERT(new_idx <= size, "Index out of bounds");
        return new_idx;
      };

  begin_iter += resolve_loc(maybe_start);

  if (maybe_end.has_value()) {
    end_iter = v.cbegin() + resolve_loc(maybe_end.value());
  }
  if (begin_iter >= end_iter) {
    return {};
  }

  if (end_iter < begin_iter) {
    end_iter = begin_iter;
  }

  std::vector<T> output(begin_iter, end_iter);
  return output;
}

} // namespace FlexFlow

#endif
