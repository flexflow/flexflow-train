#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_SLOT_NUM_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_SLOT_NUM_VALUES_H

#include "utils/slot_num_values.dtg.h"
#include "utils/singular_or_variadic.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

SlotNumValues slot_num_values_singular();
SlotNumValues slot_num_values_variadic(positive_int);

template <typename T>
SlotNumValues get_slot_num_values(SingularOrVariadic<T> const &s_or_v) {
  return s_or_v.template visit<SlotNumValues>(overload {
    [](T const &) {
      return slot_num_values_singular();
    },
    [](std::vector<T> const &v) {
      return slot_num_values_variadic(positive_int{v.size()}); 
    }
  });
}


} // namespace FlexFlow

#endif
