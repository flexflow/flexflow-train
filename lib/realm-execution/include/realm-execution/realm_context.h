
#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

enum class CopyDomain {
  SRC, // use src instance index space as copy domain (default)
  DST, // use dst instance index space as copy domain
};

/**
 * @brief An interface that wraps the rest of Realm and protects against certain
 * classes of bugs, such as shutdown bugs.
 *
 * @warning Do NOT call Realm directly unless you know what you are doing.
 */
struct RealmContext {
public:
  RealmContext(Realm::Processor processor);
  virtual ~RealmContext();

  RealmContext() = delete;
  RealmContext(RealmContext const &) = delete;
  RealmContext(RealmContext &&) = delete;

  /** \name Device mapping */
  ///\{
  Realm::Processor
      map_device_coord_to_processor(MachineSpaceCoordinate const &);
  static Realm::Memory get_nearest_memory(Realm::Processor);
  ///\}

  /** \name Current device context */
  ///\{
  Realm::Processor get_current_processor() const;
  Allocator &get_current_device_allocator();
  device_id_t get_current_device_idx() const;
  ///\}

  /** \name Task creation */
  ///\{
  Realm::Event spawn_task(Realm::Processor proc,
                          task_id_t task_id,
                          void const *args,
                          size_t arglen,
                          Realm::ProfilingRequestSet const &requests,
                          Realm::Event wait_on = Realm::Event::NO_EVENT,
                          int priority = 0);

  Realm::Event
      collective_spawn_task(Realm::Processor target_proc,
                            task_id_t task_id,
                            void const *args,
                            size_t arglen,
                            Realm::Event wait_on = Realm::Event::NO_EVENT,
                            int priority = 0);
  ///\}

  /** \name Data movement and reduction */
  ///\{
  Realm::Event
      issue_copy(ParallelTensorShape const &src_shape,
                 Realm::RegionInstance src_inst,
                 ParallelTensorShape const &dst_shape,
                 Realm::RegionInstance dst_inst,
                 Realm::ProfilingRequestSet const &requests,
                 Realm::Event wait_on = Realm::Event::NO_EVENT,
                 int priority = 0,
                 std::optional<Realm::ReductionOpID> redop_id = std::nullopt,
                 bool exclusive = false,
                 CopyDomain domain = CopyDomain::SRC);
  ///\}
  /** \name Instance management */
  ///\{
  std::pair<Realm::RegionInstance, Realm::Event>
      create_instance(Realm::Memory memory,
                      TensorShape const &shape,
                      Realm::ProfilingRequestSet const &prs,
                      Realm::Event wait_on = Realm::Event::NO_EVENT);
  ///\}

  /**
   * \brief Get the current set of outstanding events
   */
  Realm::Event get_outstanding_events();

  /**
 * \brief Create a Realm region instance with an offset index space.
 *
 * Similar to \ref create_instance, but allocates the instance with a
 * non-zero origin rect. This is used for sharded tensors where each
 * shard occupies a sub-region of the full logical tensor's index space.
 *
 * For example, given a tensor of shape [10, 16] split along dim 0
 * with degree 2:
 * - Shard 0 is allocated with rect [0..4, 0..15]
 * - Shard 1 is allocated with rect [5..9, 0..15]
 *
 * This allows plain Realm copies between shards and the combined tensor
 * to work correctly — points in each shard's index space match the
 * corresponding points in the combined tensor's index space, so Realm
 * copies data to the correct region without needing affine indirection.
 *
 * \param memory The Realm memory in which to allocate the instance.
 * \param shape The per-device tensor shape (already divided by degree).
 *              Determines the size of the instance.
 * \param offsets Per-dimension offsets into the full logical tensor.
 *                \p offsets[i] is the starting index along dimension i.
 *                For shard k along dim d with piece_size p:
 *                \p offsets[d] = k * p.
 * \param prs Realm profiling request set.
 * \param wait_on Event to wait on before creating the instance.
 * \return A pair of the created \ref Realm::RegionInstance and a
 *         \ref Realm::Event that fires when the instance is ready.
 *
 * \note The instance's index space has origin at \p offsets, not at
 *       zero. Copies to/from this instance must use its actual index
 *       space (via \c get_indexspace()) rather than a reconstructed
 *       zero-based index space.
 *
 * \see create_instance
 * \see perform_instance_allocation_for_value
 */
  std::pair<Realm::RegionInstance, Realm::Event> create_instance_with_offset(
      Realm::Memory memory,
      TensorShape const &shape,
      std::vector<int> const &offsets,
      Realm::ProfilingRequestSet const &prs,
      Realm::Event wait_on = Realm::Event::NO_EVENT);
  /**
 * \brief Create a Realm region instance wrapping an existing memory buffer.
 *
 * Used for external input tensors pre-allocated outside of Realm.
 * The instance wraps the provided pointer without copying or taking
 * ownership — the caller must ensure the buffer outlives the instance.
 *
 * \param memory The Realm memory containing the buffer.
 * \param shape The per-device tensor shape.
 * \param offsets Per-dimension offsets (for sharded tensors). Empty or
 *                all-zero for unsharded tensors.
 * \param ptr Raw pointer to the existing memory buffer.
 * \param prs Realm profiling request set.
 * \param wait_on Event to wait on before creating the instance.
 * \return Pair of the created instance and ready event.
 *
 * \note Realm takes ownership of the InstanceLayoutGeneric object but
 *       NOT of the underlying memory buffer pointed to by \p ptr.
 * \note The caller is responsible for ensuring \p ptr remains valid
 *       for the lifetime of the returned instance.
 *
 * \see create_instance
 * \see create_instance_with_offset
 */
  std::pair<Realm::RegionInstance, Realm::Event>
      create_external_instance(Realm::Memory memory,
                               TensorShape const &shape,
                               std::vector<int> const &offsets,
                               void *ptr,
                               Realm::ProfilingRequestSet const &prs,
                               Realm::Event wait_on = Realm::Event::NO_EVENT);

protected:
  /**
   * \brief Compact **and clear** the outstanding event queue
   *
   * \warning **User must block** on event or else use it, or it **will be
   * lost** (potentially resulting in a shutdown hang).
   */
  [[nodiscard]] Realm::Event merge_outstanding_events();

  void discover_machine_topology();

  static std::optional<ManagedPerDeviceFFHandle>
      make_device_handle_for_processor(Realm::Processor processor);

  /**
   * \brief Get the raw Realm runtime
   *
   * \note If you use the Realm runtime directly, you are responsible for
   * waiting on all generated events to ensure that Realm can shut down
   * correctly.
   */
  Realm::Runtime get_runtime();

private:
  Realm::Runtime runtime;
  Realm::Processor processor;
  Allocator allocator;
  std::vector<Realm::Event> outstanding_events;
  std::unordered_map<std::pair<Realm::AddressSpace, Realm::Processor::Kind>,
                     std::vector<Realm::Processor>>
      processors;
};

} // namespace FlexFlow

#endif
