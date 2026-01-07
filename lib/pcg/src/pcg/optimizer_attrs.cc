#include "pcg/optimizer_attrs.h"
#include "utils/overload.h"

namespace FlexFlow {

OptimizerAttrs
    get_optimizer_attrs_for_next_iter(OptimizerAttrs const &old_attrs) {
  if (old_attrs.has<AdamOptimizerAttrs>()) {
    AdamOptimizerAttrs old = old_attrs.get<AdamOptimizerAttrs>();
    double new_beta1_t = old.beta_t * old.beta1;
    double new_beta2_t = old.beta2_t * old.beta2;
    double new_alpha_t = old.alpha * sqrt(1 - new_beta2_t) / (1 - new_beta1_t);
    return OptimizerAttrs{AdamOptimizerAttrs{old.alpha,
                                             old.beta1,
                                             old.beta2,
                                             old.weight_decay,
                                             new_alpha_t,
                                             new_beta1_t,
                                             new_beta2_t,
                                             old.epsilon}};
  } else {
    return old_attrs;
  }
}

std::unordered_set<OptimizerSlotName>
    get_slot_names_for_optimizer(OptimizerAttrs const &attrs) {
  return attrs.visit<std::unordered_set<OptimizerSlotName>>(overload{
      [](SGDOptimizerAttrs const &sgd_attrs)
          -> std::unordered_set<OptimizerSlotName> {
        if (sgd_attrs.momentum > 0.0f) {
          return {OptimizerSlotName::SGD_V};
          ;
        } else {
          return {};
        }
      },
      [](AdamOptimizerAttrs const &) -> std::unordered_set<OptimizerSlotName> {
        return {
            OptimizerSlotName::ADAM_M,
            OptimizerSlotName::ADAM_V,
        };
      },
  });
}

} // namespace FlexFlow
