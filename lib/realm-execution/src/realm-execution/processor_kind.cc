#include "realm-execution/processor_kind.h"
#include "realm-execution/fmt/realm_processor_kind.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

DeviceType
    device_type_from_processor_kind(Realm::Processor::Kind processor_kind) {
  switch (processor_kind) {
    case Realm::Processor::Kind::LOC_PROC:
      return DeviceType::CPU;
    case Realm::Processor::Kind::TOC_PROC:
      return DeviceType::GPU;
    default:
      PANIC("Unhandled Realm::Processor::Kind", fmt::to_string(processor_kind));
  }
}

Realm::Processor::Kind processor_kind_from_device_type(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return Realm::Processor::Kind::LOC_PROC;
    case DeviceType::GPU:
      return Realm::Processor::Kind::TOC_PROC;
    default:
      PANIC("Unhandled DeviceType", device_type);
  }
}

} // namespace FlexFlow
