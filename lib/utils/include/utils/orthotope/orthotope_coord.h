#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_COORD_H

#include "utils/orthotope/orthotope_coord.dtg.h"

namespace FlexFlow {

OrthotopeCoord restrict_orthotope_coord_to_dims(
    OrthotopeCoord const &coord, std::set<nonnegative_int> const &allowed_dims);

} // namespace FlexFlow

#endif
