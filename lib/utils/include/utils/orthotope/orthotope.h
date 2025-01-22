#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_H

#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope_coord.dtg.h"

namespace FlexFlow {

nonnegative_int get_orthotope_num_dims(Orthotope const &);

nonnegative_int get_orthotope_volume(Orthotope const &);

std::unordered_set<OrthotopeCoord> get_all_coords_in_orthotope(Orthotope const &);

bool orthotope_contains_coord(Orthotope const &, OrthotopeCoord const &);

Orthotope restrict_orthotope_to_dims(Orthotope const &, std::set<nonnegative_int> const &);

nonnegative_int flatten_orthotope_coord(OrthotopeCoord const &, Orthotope const &);

OrthotopeCoord unflatten_orthotope_coord(nonnegative_int, Orthotope const &);

} // namespace FlexFlow

#endif
