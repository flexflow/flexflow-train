#include "pcg/device_id_t.h"

namespace FlexFlow {

device_id_t make_device_id_t_from_idx(nonnegative_int idx,
                                      DeviceType device_type) {
  switch (device_type) {
    case DeviceType::GPU:
      return device_id_t{gpu_id_t{idx}};
    case DeviceType::CPU:
      return device_id_t{cpu_id_t{idx}};
    default:
      PANIC("Unhandled device_type", device_type);
  }
}

} // namespace FlexFlow
