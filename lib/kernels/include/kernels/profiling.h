#ifndef _FLEXFLOW_KERNELS_PROFILING_H
#define _FLEXFLOW_KERNELS_PROFILING_H

#include "kernels/device.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/device_type.dtg.h"
#include "kernels/device_stream_t.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename F, typename... Ts>
std::optional<float>
    profiling_wrapper(F const &f, bool enable_profiling, DeviceType device_type, Ts &&...ts) {
  if (enable_profiling) {
    ProfilingSettings settings = ProfilingSettings{
      /*warmup_iters=*/0, 
      /*measure_iters=*/1,
    };
    return profiling_wrapper<F, Ts...>(f, settings, std::forward<Ts>(ts)...);
  } else {
    f(get_stream_for_device_type(device_type), std::forward<Ts>(ts)...);
    return std::nullopt;
  }
}

template <typename F, typename... Ts>
std::optional<float> profiling_wrapper(F const &f,
                                       ProfilingSettings const &settings,
                                       DeviceType device_type,
                                       Ts &&...ts) {
  ASSERT(device_type == DeviceType::GPU, 
         "Kernel profiling is currently only supported for GPUs. "
         "If you need this feature, please create an issue.");

  device_stream_t stream = get_stream_for_device_type(device_type);

  ffEvent_t t_start, t_end;
  checkCUDA(ffEventCreate(&t_start));
  checkCUDA(ffEventCreate(&t_end));

  for (int i = 0; i < settings.warmup_iters + settings.measure_iters; i++) {
    if (i == settings.warmup_iters) {
      checkCUDA(ffEventRecord(t_start, stream.require_gpu()));
    }
    f(stream, std::forward<Ts>(ts)...);
  }

  float elapsed = 0;
  checkCUDA(ffEventRecord(t_end, stream.require_gpu()));
  checkCUDA(ffEventSynchronize(t_end));
  checkCUDA(ffEventElapsedTime(&elapsed, t_start, t_end));
  checkCUDA(ffEventDestroy(t_start));
  checkCUDA(ffEventDestroy(t_end));
  return elapsed;
}

} // namespace FlexFlow

#endif
