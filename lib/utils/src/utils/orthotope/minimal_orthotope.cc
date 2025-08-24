#include "utils/orthotope/minimal_orthotope.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

MinimalOrthotope require_orthotope_is_minimal(Orthotope const &orthotope) {
  return MinimalOrthotope{
    transform(orthotope.dims,
              [](positive_int dim) { 
                return int_ge_two{dim};
              }),
  };
}

Orthotope orthotope_from_minimal_orthotope(MinimalOrthotope const &minimal_orthotope) {
  return Orthotope{
    transform(minimal_orthotope.dims,
              [](int_ge_two dim) { 
                return dim.positive_int_from_int_ge_two();
              }),
  };
}


} // namespace FlexFlow
