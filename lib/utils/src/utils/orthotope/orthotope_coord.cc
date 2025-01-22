#include "utils/orthotope/orthotope_coord.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/contains.h"

namespace FlexFlow {

OrthotopeCoord restrict_orthotope_coord_dims_to(OrthotopeCoord const &coord, std::set<nonnegative_int> const &allowed_dims) {
  return OrthotopeCoord{
    filter_idxs(coord.raw, [&](nonnegative_int idx) { return contains(allowed_dims, idx); }),
  };
}

} // namespace FlexFlow
