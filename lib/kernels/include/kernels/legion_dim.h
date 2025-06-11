#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LEGION_DIM_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LEGION_DIM_H

#include "kernels/legion_dim_t.dtg.h"
#include "kernels/legion_ordered/legion_ordered.h"
#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

legion_dim_t add_to_legion_dim(legion_dim_t legion_dim, int value);

legion_dim_t legion_dim_from_ff_dim(ff_dim_t, nonnegative_int num_dimensions);

ff_dim_t ff_dim_from_legion_dim(legion_dim_t, nonnegative_int num_dimensions);

template <typename T>
std::set<legion_dim_t> key_range(LegionOrdered<T> const &d) {
  return transform(set_of(nonnegative_range(num_elements(d))),
                   [](nonnegative_int i) { return legion_dim_t{i}; });
}

template <typename T>
FFOrdered<T>
    ff_ordered_from_legion_ordered(LegionOrdered<T> const &legion_ordered) {
  return FFOrdered<T>(legion_ordered.rbegin(), legion_ordered.rend());
}

template <typename T>
LegionOrdered<T>
    legion_ordered_from_ff_ordered(FFOrdered<T> const &ff_ordered) {
  return LegionOrdered<T>(ff_ordered.rbegin(), ff_ordered.rend());
}

} // namespace FlexFlow

#endif
