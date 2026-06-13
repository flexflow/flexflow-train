#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/overload.h"

namespace FlexFlow {

TrainingOpType training_op_attrs_get_op_type(
    TrainingOperationAttrs const &training_op_attrs) {
  return training_op_attrs.visit<TrainingOpType>(overload{
      [](PCGOperatorAttrs const &a) -> TrainingOpType {
        return TrainingOpType{
            pcg_op_attrs_get_op_type(a),
        };
      },
      [](LossAttrs const &) -> TrainingOpType {
        return TrainingOpType{TrainingOnlyOpType::LOSS};
      },
      [](CopyAttrs const &) -> TrainingOpType {
        return TrainingOpType{TrainingOnlyOpType::COPY};
      },
  });
}

} // namespace FlexFlow
