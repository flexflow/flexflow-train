#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

struct RealmContext {
public:
  RealmContext(Realm::Processor processor);
  virtual ~RealmContext();

  RealmContext() = delete;
  RealmContext(RealmContext const &) = delete;
  RealmContext(RealmContext &&) = delete;

  // Device mapping
  Realm::Processor
      map_device_coord_to_processor(MachineSpaceCoordinate const &);
  static Realm::Memory get_nearest_memory(Realm::Processor);

  // Current device context
  Realm::Processor get_current_processor() const;
  Allocator &get_current_device_allocator();
  device_handle_t const &get_current_device_handle() const;
  device_id_t get_current_device_idx() const;

  // Task creation
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

  // Instance management
  std::pair<Realm::RegionInstance, Realm::Event>
      create_instance(Realm::Memory memory,
                      TensorShape const &shape,
                      Realm::ProfilingRequestSet const &prs,
                      Realm::Event wait_on = Realm::Event::NO_EVENT);

  // Get the current set of outstanding events
  Realm::Event get_outstanding_events();

protected:
  // Compact AND CLEAR the outstanding event queue
  // Important: USER MUST BLOCK on event or else use it, or it WILL BE LOST
  [[nodiscard]] Realm::Event merge_outstanding_events();

  void discover_machine_topology();

  static std::optional<ManagedPerDeviceFFHandle>
      make_device_handle_for_processor(Realm::Processor processor);

protected:
  Realm::Runtime runtime;
  Realm::Processor processor;
  Allocator allocator;
  std::optional<ManagedPerDeviceFFHandle> managed_handle;
  device_handle_t device_handle;
  std::vector<Realm::Event> outstanding_events;
  std::unordered_map<std::pair<Realm::AddressSpace, Realm::Processor::Kind>,
                     std::vector<Realm::Processor>>
      processors;
};

} // namespace FlexFlow

#endif
