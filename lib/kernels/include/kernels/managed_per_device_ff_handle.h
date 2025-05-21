#ifndef _FLEXFLOW_KERNELS_MANAGED_HANDLE_H
#define _FLEXFLOW_KERNELS_MANAGED_HANDLE_H

#include "kernels/ff_handle.h"

namespace FlexFlow {

struct ManagedPerDeviceFFHandle {
public:
  ManagedPerDeviceFFHandle() = delete;

  explicit ManagedPerDeviceFFHandle(int num_ranks,
                                    int my_rank,
                                    size_t workSpaceSize,
                                    bool allowTensorOpMathConversion);

  ManagedPerDeviceFFHandle(ManagedPerDeviceFFHandle const &) = delete;
  ManagedPerDeviceFFHandle &
      operator=(ManagedPerDeviceFFHandle const &) = delete;

  ManagedPerDeviceFFHandle(ManagedPerDeviceFFHandle &&other) noexcept;
  ManagedPerDeviceFFHandle &
      operator=(ManagedPerDeviceFFHandle &&other) noexcept;

  ~ManagedPerDeviceFFHandle();

  PerDeviceFFHandle const &raw_handle() const;

private:
  void cleanup();

private:
  PerDeviceFFHandle *handle;
};

ManagedPerDeviceFFHandle
    initialize_single_gpu_handle(size_t workSpaceSize,
                                 bool allowTensorOpMathConversion);
ManagedPerDeviceFFHandle
    initialize_multi_gpu_handle(int num_ranks,
                                int my_rank,
                                size_t workSpaceSize,
                                bool allowTensorOpMathConversion);

} // namespace FlexFlow

#endif
