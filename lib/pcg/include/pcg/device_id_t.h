#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_DEVICE_ID_T_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_DEVICE_ID_T_H

#include "pcg/device_id_t.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

device_id_t
  make_device_id_t_from_idx(nonnegative_int idx, DeviceType device_type);


} // namespace FlexFlow

#endif
