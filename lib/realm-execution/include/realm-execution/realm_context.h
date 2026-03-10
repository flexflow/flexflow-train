#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief An interface that wraps the rest of realm and protects against certain
 * classes of bugs, such as shutdown bugs.
 *
 * @warn Do NOT call Realm directly unless you know what you are doing.
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

protected:
  /**
   * \brief Compact **and clear** the outstanding event queue
   *
   * \warning **User must block** on event or else use it, or it **will be lost**.
   */
  [[nodiscard]] Realm::Event merge_outstanding_events();

  void discover_machine_topology();

  static std::optional<ManagedPerDeviceFFHandle>
      make_device_handle_for_processor(Realm::Processor processor);

protected:
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
