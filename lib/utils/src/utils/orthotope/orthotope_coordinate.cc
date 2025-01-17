#include "utils/orthotope/orthotope_coordinate.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/exception.h"
#include "utils/orthotope/orthotope_dim_idx_t.h"
#include "utils/fmt/set.h"
#include "utils/orthotope/orthotope_dim_indexed/drop_idxs_except.h"

namespace FlexFlow {

std::set<orthotope_dim_idx_t> get_orthotope_coord_dims(OrthotopeCoordinate const &coord) {
  return coord.idxs.indices();
}

OrthotopeCoordinate orthotope_coord_drop_dims_except(OrthotopeCoordinate const &coord, std::set<orthotope_dim_idx_t> const &mask) {
  OrthotopeDimIndexed<int> new_idxs = drop_idxs_except(coord.idxs, mask);

  return OrthotopeCoordinate{new_idxs};
}

} // namespace FlexFlow
