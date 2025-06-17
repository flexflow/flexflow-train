#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_MHA_PER_DEVICE_STATE_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_MHA_PER_DEVICE_STATE_H

#include "kernels/device.h"
#include "kernels/allocation.h"
#include "kernels/ff_handle.h"
#include <memory>

namespace FlexFlow {

struct MHAPerDeviceState {
  PerDeviceFFHandle handle;
  size_t weightSize;
  size_t reserveSpaceSize;
  ffAttnDescriptor_t attnDesc;
  ffSeqDataDescriptor_t qDesc;
  ffSeqDataDescriptor_t kDesc;
  ffSeqDataDescriptor_t vDesc;
  ffSeqDataDescriptor_t oDesc;
  int *devQoSeqArray;
  int *devKvSeqArray;
  int *loWinIdx;
  int *hiWinIdx;
  void *reserveSpace;
  Allocator allocator;

  bool operator==(MHAPerDeviceState const &other) const;
  bool operator!=(MHAPerDeviceState const &other) const;

private:
  std::tuple<decltype(handle) const &,
             decltype(weightSize) const &,
             decltype(reserveSpaceSize) const &,
             decltype(attnDesc) const &,
             decltype(qDesc) const &,
             decltype(kDesc) const &,
             decltype(vDesc) const &,
             decltype(oDesc) const &,
             decltype(devQoSeqArray) const &,
             decltype(devKvSeqArray) const &,
             decltype(loWinIdx) const &,
             decltype(hiWinIdx) const &,
             decltype(reserveSpace) const &>
      tie() const;
};

FF_VISITABLE_STRUCT_NO_EQ(MHAPerDeviceState,
                          handle,
                          weightSize,
                          reserveSpaceSize,
                          attnDesc,
                          qDesc,
                          kDesc,
                          vDesc,
                          oDesc,
                          devQoSeqArray,
                          devKvSeqArray,
                          loWinIdx,
                          hiWinIdx,
                          reserveSpace,
                          allocator);

std::string format_as(MHAPerDeviceState const &x);
std::ostream &operator<<(std::ostream &s, MHAPerDeviceState const &x);


} // namespace FlexFlow

#endif
