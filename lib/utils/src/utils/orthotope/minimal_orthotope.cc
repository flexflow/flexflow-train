#include "utils/orthotope/minimal_orthotope.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/orthotope.h"

namespace FlexFlow {

nonnegative_int minimal_orthotope_get_num_dims(MinimalOrthotope const &o) {
  return num_elements(o.dims);
}

positive_int minimal_orthotope_get_volume(MinimalOrthotope const &o) {
  return orthotope_get_volume(orthotope_from_minimal_orthotope(o));
}

MinimalOrthotope require_orthotope_is_minimal(Orthotope const &orthotope) {
  return MinimalOrthotope{
      transform(orthotope.dims,
                [](positive_int dim) { return int_ge_two{dim}; }),
  };
}

Orthotope orthotope_from_minimal_orthotope(
    MinimalOrthotope const &minimal_orthotope) {
  return Orthotope{
      transform(
          minimal_orthotope.dims,
          [](int_ge_two dim) { return dim.positive_int_from_int_ge_two(); }),
  };
}

} // namespace FlexFlow
