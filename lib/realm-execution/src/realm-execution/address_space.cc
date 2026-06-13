#include "realm-execution/address_space.h"

namespace FlexFlow {

node_idx_t node_idx_from_realm_address_space(Realm::AddressSpace address_space) {
  return node_idx_t{
    nonnegative_int{
      static_cast<size_t>(address_space),
    },
  };
}

Realm::AddressSpace realm_address_space_from_node_idx(node_idx_t node_idx) {
  return static_cast<Realm::AddressSpace>(node_idx.raw.unwrap_nonnegative());
};

} // namespace FlexFlow
