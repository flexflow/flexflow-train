#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "local-execution/ops/attention.h"
#include "local-execution/ops/batch_matmul.h"
#include "local-execution/ops/batch_norm.h"
#include "local-execution/ops/broadcast.h"
#include "local-execution/ops/cast.h"
#include "local-execution/ops/combine.h"
#include "local-execution/ops/concat.h"
#include "local-execution/ops/conv_2d.h"
#include "local-execution/ops/dropout.h"
#include "local-execution/ops/element_binary.h"
#include "local-execution/ops/element_unary.h"
#include "local-execution/ops/embedding.h"
#include "local-execution/ops/flat.h"
#include "local-execution/ops/gather.h"
#include "local-execution/ops/input.h"
#include "local-execution/ops/layer_norm.h"
#include "local-execution/ops/linear.h"
#include "local-execution/ops/noop.h"
#include "local-execution/ops/pool_2d.h"
#include "local-execution/ops/reduce.h"
#include "local-execution/ops/reduction.h"
#include "local-execution/ops/repartition.h"
#include "local-execution/ops/replicate.h"
#include "local-execution/ops/reshape.h"
#include "local-execution/ops/reverse.h"
#include "local-execution/ops/softmax.h"
#include "local-execution/ops/split.h"
#include "local-execution/ops/topk.h"
#include "local-execution/ops/transpose.h"
#include "utils/record_formatter.h"
#include "utils/variant.h"
#include <variant>

namespace FlexFlow {

std::vector<ParallelTensorShape> get_output_shapes(
    PCGOperatorAttrs const &op_params,
    std::vector<ParallelTensorShape> const &input_tensor_shapes);

bool is_valid(PCGOperatorAttrs const &,
              std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
