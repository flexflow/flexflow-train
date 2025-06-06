#include "kernels/legion_dim.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
template std::set<legion_dim_t> key_range(LegionOrdered<T> const &);

legion_dim_t add_to_legion_dim(legion_dim_t legion_dim, int value) {
  return legion_dim_t{
      nonnegative_int{legion_dim.value.unwrap_nonnegative() + value}};
}

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim,
                                    nonnegative_int num_dimensions) {
  return legion_dim_t{nonnegative_int{num_dimensions.unwrap_nonnegative() -
                                      ff_dim.value.unwrap_nonnegative() - 1}};
  ;
}

} // namespace FlexFlow
