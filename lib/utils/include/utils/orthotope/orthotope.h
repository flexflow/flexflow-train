#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_H

#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope_coordinate.dtg.h"
#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include <unordered_set>

namespace FlexFlow {

std::set<orthotope_dim_idx_t> get_orthotope_dims(Orthotope const &);
int orthotope_num_dims(Orthotope const &);

bool orthotope_contains_coord(Orthotope const &, OrthotopeCoordinate const &);
std::unordered_set<OrthotopeCoordinate> orthotope_get_contained_coordinates(Orthotope const &);

int orthotope_get_volume(Orthotope const &);

Orthotope orthotope_drop_dims_except(Orthotope const &, std::set<orthotope_dim_idx_t> const &);

} // namespace FlexFlow

#endif
