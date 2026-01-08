#include "substitutions/unlabelled/pattern_node_output.h"

namespace FlexFlow {

PatternNode get_src_node(PatternNodeOutput const &o) {
  return PatternNode{o.raw_dataflow_output.node};
}

TensorSlotName get_slot_name(PatternNodeOutput const &o) {
  return o.raw_dataflow_output.slot_name;
}

} // namespace FlexFlow
