#include "utils/orthotope/orthotope_dim_indexed/all_of.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

bool all_of(OrthotopeDimIndexed<bool> const &d) {
  return all_of(d.get_contents());
}

} // namespace FlexFlow
