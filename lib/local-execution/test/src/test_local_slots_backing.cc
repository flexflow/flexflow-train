#include "kernels/attention_kernels.h"
#include "kernels/local_cpu_allocator.h"
#include "local-execution/local_cost_estimator.h"
#include "local-execution/local_slots_backing.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/variant.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test_utils.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalSlotsBacking -- Attention Op") {
    // allocate input memory
    Allocator allocator = create_local_cpu_memory_allocator();
    nonnegative_int embed_dim = 32_n;
    nonnegative_int num_heads = 10_n;

    nonnegative_int batch_size = 40_n;
    nonnegative_int seq_len = 48_n;
    nonnegative_int feature_size = 36_n;

    DataType dtype = DataType::FLOAT;
    TensorShape input_tensor_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{batch_size, seq_len, feature_size}},
        DataType::FLOAT,
    };
    TensorShape query_shape = input_tensor_shape;
    TensorShape key_shape = input_tensor_shape;
    TensorShape value_shape = input_tensor_shape;
    GenericTensorAccessorW query = allocator.allocate_tensor(query_shape);
    GenericTensorAccessorW key = allocator.allocate_tensor(key_shape);
    GenericTensorAccessorW value = allocator.allocate_tensor(value_shape);

    // build graph
    ComputationGraphBuilder cg_builder;
    tensor_guid_t query_guid =
        cg_builder.create_input(query_shape, CreateGrad::YES);
    tensor_guid_t key_guid =
        cg_builder.create_input(key_shape, CreateGrad::YES);
    tensor_guid_t value_guid =
        cg_builder.create_input(value_shape, CreateGrad::YES);

    std::string layer_name = "attn1";
    tensor_guid_t output_guid =
        cg_builder.multihead_attention(query_guid,
                                       key_guid,
                                       value_guid,
                                       embed_dim,
                                       num_heads,
                                       /*kdim=*/embed_dim,
                                       /*vdim=*/embed_dim,
                                       /*dropout=*/0.0f,
                                       /*bias=*/true,
                                       /*add_bias_kv=*/false,
                                       /*add_zero_attn=*/false,
                                       /*initializer=*/std::nullopt,
                                       /*maybe_name=*/layer_name);

    layer_guid_t layer_guid =
        get_layer_by_name(cg_builder.computation_graph, layer_name);

    TensorBackingMap tensor_backing_map = {
        {query_guid, query}, {key_guid, key}, {value_guid, value}};

    // runtime arg config
    ProfilingSettings settings = ProfilingSettings{/*warmup_iters=*/0,
                                                   /*measure_iters=*/0};
    PerDeviceFFHandle handle = get_mock_per_device_ff_handle();
    RuntimeArgConfig runtime_arg_config =
        RuntimeArgConfig{DeviceSpecific<PerDeviceFFHandle>::create(handle),
                         EnableProfiling::NO,
                         settings};

    LocalSlotsBacking local_slots_backing = {tensor_backing_map,
                                             runtime_arg_config};

    SUBCASE("LocalSlotsBacking::allocate_outgoing_tensors") {
      auto get_result_shape_and_dtype_for_tensor_guid_and_map =
          [&](tensor_guid_t t,
              TensorBackingMap m) -> std::pair<ArrayShape, DataType> {
        GenericTensorAccessorW accessor = m.at(t);
        return get_shape_and_datatype(accessor);
      };

      SUBCASE("Input (QKV) and gradient tensors allocation") {

        // allocate all tensors from input nodes
        for (layer_guid_t const &node :
             topological_ordering(cg_builder.computation_graph)) {
          if (node == layer_guid) {
            break;
          }
          local_slots_backing.allocate_outgoing_tensors(
              node, cg_builder.computation_graph, allocator);
        }

        SUBCASE("Query grad") {
          std::pair<ArrayShape, DataType> result =
              get_result_shape_and_dtype_for_tensor_guid_and_map(
                  query_guid, local_slots_backing.gradient_tensor_mapping);
          std::pair<ArrayShape, DataType> correct = {
              array_shape_from_tensor_shape(query_shape), dtype};
          CHECK(result == correct);
        }
        SUBCASE("Key grad") {
          std::pair<ArrayShape, DataType> result =
              get_result_shape_and_dtype_for_tensor_guid_and_map(
                  key_guid, local_slots_backing.gradient_tensor_mapping);
          std::pair<ArrayShape, DataType> correct = {
              array_shape_from_tensor_shape(key_shape), dtype};
          CHECK(result == correct);
        }
        SUBCASE("Value grad") {
          std::pair<ArrayShape, DataType> result =
              get_result_shape_and_dtype_for_tensor_guid_and_map(
                  value_guid, local_slots_backing.gradient_tensor_mapping);
          std::pair<ArrayShape, DataType> correct = {
              array_shape_from_tensor_shape(value_shape), dtype};
          CHECK(result == correct);
        }
      }
      SUBCASE("Output and gradient tensors allocation") {
        local_slots_backing.allocate_outgoing_tensors(
            layer_guid, cg_builder.computation_graph, allocator);
        SUBCASE("Output") {
          std::pair<ArrayShape, DataType> result =
              get_result_shape_and_dtype_for_tensor_guid_and_map(
                  output_guid, local_slots_backing.tensor_mapping);
          std::pair<ArrayShape, DataType> correct = {
              array_shape_from_tensor_shape(
                  get_tensor_attrs(cg_builder.computation_graph, output_guid)
                      .shape),
              dtype};
          CHECK(result == correct);
        }
        SUBCASE("Output grad") {
          std::pair<ArrayShape, DataType> result =
              get_result_shape_and_dtype_for_tensor_guid_and_map(
                  output_guid, local_slots_backing.gradient_tensor_mapping);
          std::pair<ArrayShape, DataType> correct = {
              array_shape_from_tensor_shape(
                  get_tensor_attrs(cg_builder.computation_graph, output_guid)
                      .shape),
              dtype};
          CHECK(result == correct);
        }
      }

      SUBCASE("Tensor slots") {
        local_slots_backing.allocate_outgoing_tensors(
            layer_guid, cg_builder.computation_graph, allocator);
        SUBCASE("Input tensor slots") {
          std::vector<tensor_guid_t> correct_incoming_tensors =
              get_incoming_tensors(cg_builder.computation_graph, layer_guid);
          CHECK(correct_incoming_tensors ==
                local_slots_backing.input_tensor_slots.at(layer_guid));
        }
        SUBCASE("Output tensor slots") {
          std::vector<tensor_guid_t> correct_outgoing_tensors =
              get_outgoing_tensors(cg_builder.computation_graph, layer_guid);
          CHECK(correct_outgoing_tensors ==
                local_slots_backing.output_tensor_slots.at(layer_guid));
        }
      }
    }

    SUBCASE("Construct Slots Backings") {
      enum Slots {
        QUERY,
        KEY,
        VALUE,
        WEIGHTS,
        OUTPUT,
        QUERY_PARALLEL_TENSOR_SHAPE,
        QPROJSIZE,
        ATTRS,
        PROFILING,
        HANDLE,
      };
      MultiHeadAttentionAttrs attrs =
          get_layer_attrs(cg_builder.computation_graph, layer_guid)
              .op_attrs.get<MultiHeadAttentionAttrs>();
      OpTaskBinding binding = [&] {
        OpTaskBinding b;
        b.bind(QUERY, input_tensor(0));
        b.bind(KEY, input_tensor(1));
        b.bind(VALUE, input_tensor(2));
        b.bind(WEIGHTS, weight_tensor(3));
        b.bind(OUTPUT, output_tensor(0));

        b.bind_grad(QUERY, input_tensor(0));

        b.bind_arg(QPROJSIZE, get_qProjSize(attrs));
        b.bind_arg(ATTRS, attrs);
        b.bind_arg(QUERY_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(0));
        b.bind_arg(PROFILING, profiling_settings());
        b.bind_arg(HANDLE, ff_handle());
        return b;
      }();

      // allocate all incoming and outgoing tensors for graph
      for (layer_guid_t const &node :
           topological_ordering(cg_builder.computation_graph)) {
        local_slots_backing.allocate_outgoing_tensors(
            node, cg_builder.computation_graph, allocator);
      }

      SUBCASE("LocalSlotsBacking::construct_tensor_slots_backing") {
        TensorSlotsBackingWithoutAddresses result =
            get_slots_backing_without_tensor_allocation_addresses(
                local_slots_backing.construct_tensor_slots_backing(binding,
                                                                   layer_guid));
        TensorSlotsBackingWithoutAddresses correct = [&] {
          TensorShape weights_shape = throw_if_unexpected(
              get_weights_shape(attrs, query_shape, key_shape, value_shape));
          GenericTensorAccessorW weights =
              allocator.allocate_tensor(weights_shape);

          TensorAttrs output_attrs =
              get_tensor_attrs(cg_builder.computation_graph, output_guid);
          GenericTensorAccessorW output =
              allocator.allocate_tensor(output_attrs.shape);
          return get_slots_backing_without_tensor_allocation_addresses(
              TensorSlotsBacking{
                  {SlotGradId{slot_id_t{QUERY}, IsGrad::NO}, query},
                  {SlotGradId{slot_id_t{KEY}, IsGrad::NO}, key},
                  {SlotGradId{slot_id_t{VALUE}, IsGrad::NO}, value},
                  {SlotGradId{slot_id_t{WEIGHTS}, IsGrad::NO}, weights},
                  {SlotGradId{slot_id_t{OUTPUT}, IsGrad::NO}, output},
                  {SlotGradId{slot_id_t{QUERY}, IsGrad::YES}, query}});
        }();

        CHECK(result == correct);
      }
      SUBCASE("LocalSlotsBacking::construct_arg_slots_backing") {
        ArgSlotsBacking result =
            local_slots_backing.construct_arg_slots_backing(binding,
                                                            layer_guid);

        ArgSlotsBacking correct = [&] {
          ParallelTensorShape query_parallel_tensor_shape =
              lift_to_parallel(query_shape);

          return ArgSlotsBacking{
              {slot_id_t{QPROJSIZE},
               ConcreteArgSpec::create(get_qProjSize(attrs))},
              {slot_id_t{ATTRS}, ConcreteArgSpec::create(attrs)},
              {slot_id_t{QUERY_PARALLEL_TENSOR_SHAPE},
               ConcreteArgSpec::create(query_parallel_tensor_shape)},
              {slot_id_t{PROFILING},
               ConcreteArgSpec::create(runtime_arg_config.profiling_settings)},
              {slot_id_t{HANDLE}, ConcreteArgSpec::create(handle)}};
        }();

        CHECK(result == correct);
      }

      SUBCASE("LocalSlotsBacking::resolve_runtime_arg_ref_spec") {
        RuntimeArgRefSpec ref_spec = RuntimeArgRefSpec::create(ff_handle());
        ConcreteArgSpec arg_spec =
            local_slots_backing.resolve_runtime_arg_ref_spec(ref_spec);

        PerDeviceFFHandle result_handle = arg_spec.get<PerDeviceFFHandle>();
        CHECK(result_handle == handle);
      }
    }
  }
}
