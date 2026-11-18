#ifndef _FLEXFLOW_BIN_IMPORT_PT2_INCLUDE_IMPORT_PT2_PT2_TO_COMPUTATION_GRAPH_H
#define _FLEXFLOW_BIN_IMPORT_PT2_INCLUDE_IMPORT_PT2_PT2_TO_COMPUTATION_GRAPH_H

#include "pcg/computation_graph.dtg.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace FlexFlow {

/**
 * @brief Thrown when a .pt2 graph uses something the importer does not
 * support (an unknown ATen op, dynamic shapes, unsupported arguments, ...).
 */
struct Pt2ImportError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Rebuild a serialized torch.export ExportedProgram (the contents of
 * <tt>models/model.json</tt> in a .pt2 archive) as a FlexFlow
 * ComputationGraph, using ComputationGraphBuilder.
 *
 * Each ATen node becomes the matching builder call. Parameters and buffers
 * are not imported as tensors; they are implied by the op that owns them (e.g.
 * <tt>aten.conv2d</tt>'s weight determines the conv's output channels and
 * kernel size, and its bias determines <tt>use_bias</tt>). Weight values are
 * not imported.
 *
 * After every op, the shape FlexFlow infers is checked against the shape
 * recorded in the .pt2. Any mismatch throws Pt2ImportError naming the node.
 */
ComputationGraph
    computation_graph_from_pt2_model_json(nlohmann::json const &model);

} // namespace FlexFlow

#endif
