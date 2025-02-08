#include "substitutions/operator_pattern/eval_list_access.h"
#include "substitutions/operator_pattern/get_attribute.h"
#include "utils/containers/at_idx.h"
#include "utils/containers/make.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<OperatorAttributeValue>
    eval_list_access(PCGOperatorAttrs const &attrs,
                     OperatorAttributeListIndexAccess const &acc) {
  std::optional<OperatorAttributeValue> from_attr =
      get_attribute(attrs, acc.attribute_key);

  if (!from_attr.has_value()) {
    return std::nullopt;
  }

  return from_attr.value().visit<std::optional<OperatorAttributeValue>>(
      [&](auto const &v) -> std::optional<OperatorAttributeValue> {
        using T = std::decay_t<decltype(v)>;

        if constexpr (std::is_same_v<T, std::vector<nonnegative_int>>) {
          return transform(at_idx(v, acc.index),
                           make<OperatorAttributeValue>());
        } else if constexpr (std::is_same_v<T, std::vector<ff_dim_t>>) {
          return transform(at_idx(v, acc.index),
                           make<OperatorAttributeValue>());
        } else {
          throw mk_runtime_error("Invalid operand");
        }
      });
}

} // namespace FlexFlow
