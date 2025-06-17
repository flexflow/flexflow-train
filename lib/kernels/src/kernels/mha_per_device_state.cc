#include "kernels/mha_per_device_state.h"

namespace FlexFlow {

bool MHAPerDeviceState::operator==(MHAPerDeviceState const &other) const {
  return this->tie() == other.tie();
}

bool MHAPerDeviceState::operator!=(MHAPerDeviceState const &other) const {
  return this->tie() != other.tie();
}

std::
    tuple<PerDeviceFFHandle const &, size_t const &, size_t const &, ffAttnDescriptor_t const &, ffSeqDataDescriptor_t const &, ffSeqDataDescriptor_t const &, ffSeqDataDescriptor_t const &, ffSeqDataDescriptor_t const &, int *const &, int *const &, int *const &, int *const &, void *const &>
    MHAPerDeviceState::tie() const {
  return std::tie(this->handle,
                  this->weightSize,
                  this->reserveSpaceSize,
                  this->attnDesc,
                  this->qDesc,
                  this->kDesc,
                  this->vDesc,
                  this->oDesc,
                  this->devQoSeqArray,
                  this->devKvSeqArray,
                  this->loWinIdx,
                  this->hiWinIdx,
                  this->reserveSpace);
}

std::string format_as(MHAPerDeviceState const &x) {
  return fmt::format("MHAPerDeviceState");
}

std::ostream &operator<<(std::ostream &s, MHAPerDeviceState const &x) {
  return (s << fmt::to_string(x));
}


} // namespace FlexFlow
