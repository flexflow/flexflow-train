#include "utils/orthotope/orthotope_coord.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_idxs.h"

namespace FlexFlow {

nonnegative_int orthotope_coord_num_dims(OrthotopeCoord const &coord) {
  return num_elements(coord.raw);
}

OrthotopeCoord restrict_orthotope_coord_to_dims(
    OrthotopeCoord const &coord,
    std::set<nonnegative_int> const &allowed_dims) {
  return OrthotopeCoord{
      filter_idxs(
          coord.raw,
          [&](nonnegative_int idx) { return contains(allowed_dims, idx); }),
  };
}

} // namespace FlexFlow
