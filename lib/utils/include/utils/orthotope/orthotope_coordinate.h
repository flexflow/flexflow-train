#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_COORDINATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_COORDINATE_H

#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include "utils/orthotope/orthotope_coordinate.dtg.h"
#include <set>

namespace FlexFlow {

std::set<orthotope_dim_idx_t> get_orthotope_coord_dims(OrthotopeCoordinate const &);

OrthotopeCoordinate orthotope_coord_drop_dims_except(OrthotopeCoordinate const &, std::set<orthotope_dim_idx_t> const &);

} // namespace FlexFlow

#endif
