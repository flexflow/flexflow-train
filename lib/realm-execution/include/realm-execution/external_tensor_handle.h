#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_EXTERNAL_TENSOR_HANDLE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_EXTERNAL_TENSOR_HANDLE_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "realm-execution/realm.h"

namespace FlexFlow {

/**
 * \brief A handle to an externally-allocated tensor for use as a
 * pre-allocated input to \ref create_pcg_instance.
 *
 * Memory is allocated in Z_COPY (zero-copy pinned) memory when available,
 * falling back to SYSTEM_MEM. This ensures the buffer is writable from CPU
 * and accessible from GPU via Realm copies.
 *
 * \note The handle must outlive the PCGInstance that uses it.
 * \note Realm takes ownership of the instance layout but NOT the buffer.
 *
 * \see RealmContext::create_external_tensor
 */
struct ExternalTensorHandle {
  ExternalTensorHandle() = delete;

  float *get_float_ptr() const;
  double *get_double_ptr() const;
  void *get_ptr() const;

  TensorShape shape;
  Realm::RegionInstance instance;
  Realm::Event ready;

private:
  friend struct RealmContext;

  ExternalTensorHandle(TensorShape const &shape,
                       Realm::RegionInstance instance,
                       Realm::Event ready,
                       Allocator allocator,
                       GenericTensorAccessorW accessor);

  Allocator allocator;
  GenericTensorAccessorW accessor;
};

} // namespace FlexFlow

#endif
