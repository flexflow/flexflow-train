#include "utils/graph/instances/unordered_set_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/json.h"
#include "utils/graph/labelled_dataflow_graph/json.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/json.h"
#include "utils/graph/labelled_open_dataflow_graph/json.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/json.h"

namespace FlexFlow {

template class UnorderedSetKwargDataflowGraph<int>;

} // namespace FlexFlow

namespace nlohmann {

template struct adl_serializer<::FlexFlow::KwargDataflowGraph<int>>;
template struct adl_serializer<::FlexFlow::LabelledDataflowGraph<int, int>>;
template struct adl_serializer<
    ::FlexFlow::LabelledKwargDataflowGraph<int, int, int>>;
template struct adl_serializer<::FlexFlow::LabelledOpenDataflowGraph<int, int>>;
template struct adl_serializer<
    ::FlexFlow::LabelledOpenKwargDataflowGraph<int, int, int, int>>;

} // namespace nlohmann
