#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/device_type.dtg.h"
#include "task-spec/permissions.h"
#include "utils/exception.h"

namespace FlexFlow {

static DeviceType infer_device_type_from_memory_and_processor(
    Realm::Memory inst_memory, Realm::Processor for_processor) {
  DeviceType device_type;
  switch (inst_memory.kind()) {
    case Realm::Memory::SYSTEM_MEM:
      // Only accessible on CPU
      device_type = DeviceType::CPU;
      break;
    case Realm::Memory::GPU_FB_MEM:
      // Only accessible on GPU
      device_type = DeviceType::GPU;
      break;
    case Realm::Memory::Z_COPY_MEM: {
      // Accessible on either CPU or GPU, so infer based on where we're trying
      // to access from
      switch (for_processor.kind()) {
        case Realm::Processor::LOC_PROC:
          device_type = DeviceType::CPU;
          break;
        case Realm::Processor::TOC_PROC:
          device_type = DeviceType::GPU;
          break;
        default:
          PANIC("Unexpected Realm Processor kind", for_processor.kind());
      }
    } break;
    default:
      PANIC("Unexpected Realm Memory kind", inst_memory.kind());
  }
  return device_type;
}

DynamicTensorAccessor dynamic_tensor_accessor_from_instance(
    Realm::RegionInstance inst,
    Realm::Event ready,
    ParallelTensorShape const &parallel_tensor_shape,
    Permissions const &permissions,
    Realm::Processor for_processor) {
  ready.wait();

  DeviceType device_type = infer_device_type_from_memory_and_processor(
      inst.get_location(), for_processor);

  TensorShape per_device_shape =
      get_per_device_shape(parallel_tensor_shape); // ← was get_piece_shape

  size_t expected_size = static_cast<size_t>(
      static_cast<int>(get_size_in_bytes(per_device_shape).unwrap_num_bytes()));

  void *ptr = inst.pointer_untyped(/*offset=*/0, /*datalen=*/expected_size);

  if (permissions == Permissions::RO) {
    return DynamicTensorAccessor{GenericTensorAccessorR{
        per_device_shape, ptr, device_type}}; // ← was get_piece_shape
  } else {
    return DynamicTensorAccessor{GenericTensorAccessorW{
        per_device_shape, ptr, device_type}}; // ← was get_piece_shape
  }
}
} // namespace FlexFlow
