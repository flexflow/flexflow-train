#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenDataflowValue isomorphism_map_r_open_dataflow_value_from_l(
    OpenDataflowGraphIsomorphism const &iso, OpenDataflowValue const &l_value) {
  return l_value.visit<OpenDataflowValue>(overload{
      [&](DataflowGraphInput const &l_input) {
        return OpenDataflowValue{
            iso.input_mapping.at_l(l_input),
        };
      },
      [&](DataflowOutput const &l_output) {
        return OpenDataflowValue{
            isomorphism_map_r_dataflow_output_from_l(iso, l_output),
        };
      },
  });
}

OpenDataflowValue isomorphism_map_l_open_dataflow_value_from_r(
    OpenDataflowGraphIsomorphism const &iso, OpenDataflowValue const &r_value) {
  return r_value.visit<OpenDataflowValue>(overload{
      [&](DataflowGraphInput const &r_input) {
        return OpenDataflowValue{
            iso.input_mapping.at_r(r_input),
        };
      },
      [&](DataflowOutput const &r_output) {
        return OpenDataflowValue{
            isomorphism_map_l_dataflow_output_from_r(iso, r_output),
        };
      },
  });
}

DataflowOutput isomorphism_map_r_dataflow_output_from_l(
    OpenDataflowGraphIsomorphism const &iso, DataflowOutput const &l_output) {
  return DataflowOutput{
      iso.node_mapping.at_l(l_output.node),
      l_output.idx,
  };
}

DataflowOutput isomorphism_map_l_dataflow_output_from_r(
    OpenDataflowGraphIsomorphism const &iso, DataflowOutput const &r_output) {
  return DataflowOutput{
      iso.node_mapping.at_r(r_output.node),
      r_output.idx,
  };
}

} // namespace FlexFlow
