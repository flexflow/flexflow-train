#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FF_HANDLE_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FF_HANDLE_H

#ifdef FF_USE_NCCL
#include <nccl.h>
#endif

#include "kernels/device.h"

namespace FlexFlow {

struct PerDeviceFFHandle {
public:
  ffHandle_t dnn;
  ffblasHandle_t blas;

  void *workSpace;
  size_t workSpaceSize;
  bool allowTensorOpMathConversion;

#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
#endif
};

std::string format_as(PerDeviceFFHandle const &x);
std::ostream &operator<<(std::ostream &s, PerDeviceFFHandle const &x);

} // namespace FlexFlow

#endif
