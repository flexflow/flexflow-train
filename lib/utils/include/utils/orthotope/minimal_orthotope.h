#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_ORTHOTOPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_ORTHOTOPE_H

#include "utils/orthotope/minimal_orthotope.dtg.h"
#include "utils/orthotope/orthotope.dtg.h"

namespace FlexFlow {

nonnegative_int minimal_orthotope_get_num_dims(MinimalOrthotope const &);
positive_int minimal_orthotope_get_volume(MinimalOrthotope const &);

MinimalOrthotope require_orthotope_is_minimal(Orthotope const &);
Orthotope orthotope_from_minimal_orthotope(MinimalOrthotope const &);

} // namespace FlexFlow

#endif
