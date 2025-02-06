#include "substitutions/operator_pattern/get_attribute_map.h"
#include "substitutions/operator_pattern/get_attribute.h"
#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_key.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"

namespace FlexFlow {

std::unordered_map<OperatorAttributeKey, OperatorAttributeValue>
    get_attribute_map(PCGOperatorAttrs const &op_attrs) {
  std::unordered_map<OperatorAttributeKey, OperatorAttributeValue> result;

  for (OperatorAttributeKey const &attr_key : all_operator_attribute_keys()) {
    std::optional<OperatorAttributeValue> attr_value =
        get_attribute(op_attrs, attr_key);

    if (attr_value.has_value()) {
      result.insert({attr_key, attr_value.value()});
    }
  }

  return result;
}

} // namespace FlexFlow
