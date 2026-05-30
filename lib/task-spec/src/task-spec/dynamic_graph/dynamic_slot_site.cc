#include "task-spec/dynamic_graph/dynamic_slot_site.h"
#include "utils/overload.h"

namespace FlexFlow {

DynamicValueAttrs
    dynamic_value_attrs_for_slot_site(DynamicSlotSite const &slot) {
  return slot.visit<DynamicValueAttrs>(overload{

      [](ExternalDynamicSlotSite const &external_slot) -> DynamicValueAttrs {
        return external_slot.value;
      },

      [](InternalDynamicSlotSite const &internal_slot) -> DynamicValueAttrs {
        switch (internal_slot.direction) {
          case TensorDirection::INCOMING:
            return internal_slot.invocation.inputs.at(internal_slot.slot_name);
          case TensorDirection::OUTPUT:
            return internal_slot.invocation.outputs.at(internal_slot.slot_name);
          default:
            PANIC("Unexpected direction {}", internal_slot.direction);
        }
      }});
}

} // namespace FlexFlow
