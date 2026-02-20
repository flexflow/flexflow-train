#include "substitutions/unlabelled/standard_pattern_edge.h"

namespace FlexFlow {

PatternNode get_src_node(StandardPatternEdge const &e) {
  return PatternNode{e.raw_edge.src.node};
}

PatternNode get_dst_node(StandardPatternEdge const &e) {
  return PatternNode{e.raw_edge.dst.node};
}

TensorSlotName get_src_slot_name(StandardPatternEdge const &e) {
  return e.raw_edge.src.slot_name;
}

TensorSlotName get_dst_slot_name(StandardPatternEdge const &e) {
  return e.raw_edge.dst.slot_name;
}

} // namespace FlexFlow
