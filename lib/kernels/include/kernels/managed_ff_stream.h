#ifndef _FLEXFLOW_KERNELS_MANAGED_FF_STREAM_H
#define _FLEXFLOW_KERNELS_MANAGED_FF_STREAM_H

#include "kernels/device.h"

namespace FlexFlow {

struct ManagedFFStream {
public:
  ManagedFFStream();

  ManagedFFStream(ManagedFFStream const &) = delete;
  ManagedFFStream &operator=(ManagedFFStream const &) = delete;

  ManagedFFStream(ManagedFFStream &&other) noexcept;
  ManagedFFStream &operator=(ManagedFFStream &&other) noexcept;

  ~ManagedFFStream();

  ffStream_t const &raw_stream() const;

private:
  void cleanup();

private:
  ffStream_t *stream;
};

} // namespace FlexFlow

#endif
