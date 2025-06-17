#include <doctest/doctest.h>
#include "kernels/local_cuda_allocator.h"
#include "local-execution/local_cost_estimator.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/local_task_registry.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/task_signature_impl.h"
#include "utils/fmt/optional.h"
#include "utils/fmt/unordered_map.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalTaskRegistry") {
    layer_guid_t layer_guid = layer_guid_t{Node{0}};
    positive_int embed_dim = 32_p;
    positive_int num_heads = 10_p;
    ComputationGraphOpAttrs attrs =
        ComputationGraphOpAttrs{MultiHeadAttentionAttrs{
            /*embed_dim=*/embed_dim,
            /*num_heads=*/num_heads,
            /*kdim=*/embed_dim,
            /*vdim=*/embed_dim,
            /*dropout=*/0.0,
            /*bias=*/true,
            /*add_bias_kv=*/false,
            /*add_zero_attn=*/false,
        }};

    SUBCASE("register single layer") {
      LocalTaskRegistry task_registry = construct_local_task_registry_for_layers(
          {{layer_guid, LayerAttrs{attrs, std::nullopt}}});

      LocalTaskRegistry correct_task_registry = [&] {
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            init_task_ids = {{layer_guid, task_id_t::ATTENTION_INIT_TASK_ID}};
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            fwd_task_ids = {{layer_guid, task_id_t::ATTENTION_FWD_TASK_ID}};
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            bwd_task_ids = {{layer_guid, task_id_t::ATTENTION_BWD_TASK_ID}};
        std::unordered_map<task_id_t, TaskSignatureAndImpl> task_mapping = {
            {task_id_t::ATTENTION_INIT_TASK_ID,
             get_task_sig_impl(task_id_t::ATTENTION_INIT_TASK_ID)},
            {task_id_t::ATTENTION_FWD_TASK_ID,
             get_task_sig_impl(task_id_t::ATTENTION_FWD_TASK_ID)},
            {task_id_t::ATTENTION_BWD_TASK_ID,
             get_task_sig_impl(task_id_t::ATTENTION_BWD_TASK_ID)}};
        return LocalTaskRegistry{
            init_task_ids, fwd_task_ids, bwd_task_ids, task_mapping};
      }();

      CHECK(task_registry == correct_task_registry);
    }

    SUBCASE("multiple layers same task") {
      layer_guid_t other_layer_guid = layer_guid_t{Node{1}};
      LocalTaskRegistry task_registry = construct_local_task_registry_for_layers({
          {layer_guid, LayerAttrs{attrs, std::nullopt}},
          {other_layer_guid, LayerAttrs{attrs, std::nullopt}},
      });

      SUBCASE("layer to task ids") {
        std::unordered_map<layer_guid_t, std::optional<task_id_t>> correct = {
            {layer_guid, task_id_t::ATTENTION_INIT_TASK_ID},
            {other_layer_guid, task_id_t::ATTENTION_INIT_TASK_ID},
        };
        CHECK(correct == task_registry.init_task_ids);
      }

      SUBCASE("task to signature+impl mapping") {
        std::unordered_map<task_id_t, TaskSignatureAndImpl>
            correct_task_mapping = {
                {task_id_t::ATTENTION_INIT_TASK_ID,
                 get_task_sig_impl(task_id_t::ATTENTION_INIT_TASK_ID)},
                {task_id_t::ATTENTION_FWD_TASK_ID,
                 get_task_sig_impl(task_id_t::ATTENTION_FWD_TASK_ID)},
                {task_id_t::ATTENTION_BWD_TASK_ID,
                 get_task_sig_impl(task_id_t::ATTENTION_BWD_TASK_ID)}};
        CHECK(correct_task_mapping == task_registry.task_mapping);
      }
    }

    SUBCASE("different attrs, still same task fn mapping") {
      layer_guid_t layer_1 = layer_guid_t{Node{1}};
      positive_int embed_dim = 100_p;
      layer_guid_t layer_2 = layer_guid_t{Node{2}};
      ComputationGraphOpAttrs other_attrs =
          ComputationGraphOpAttrs{MultiHeadAttentionAttrs{
              /*embed_dim=*/embed_dim,
              /*num_heads=*/num_heads,
              /*kdim=*/embed_dim,
              /*vdim=*/embed_dim,
              /*dropout=*/0.0,
              /*bias=*/true,
              /*add_bias_kv=*/false,
              /*add_zero_attn=*/false,
          }};
      LocalTaskRegistry task_registry = construct_local_task_registry_for_layers({
          {layer_guid, LayerAttrs{attrs, std::nullopt}},
          {layer_1, LayerAttrs{attrs, std::nullopt}},
          {layer_2, LayerAttrs{other_attrs, std::nullopt}},
      });

      std::unordered_map<task_id_t, TaskSignatureAndImpl> correct_task_mapping =
          {{task_id_t::ATTENTION_INIT_TASK_ID,
            get_task_sig_impl(task_id_t::ATTENTION_INIT_TASK_ID)},
           {task_id_t::ATTENTION_FWD_TASK_ID,
            get_task_sig_impl(task_id_t::ATTENTION_FWD_TASK_ID)},
           {task_id_t::ATTENTION_BWD_TASK_ID,
            get_task_sig_impl(task_id_t::ATTENTION_BWD_TASK_ID)}};

      CHECK(correct_task_mapping == task_registry.task_mapping);
    }

    SUBCASE("equality") {
      SUBCASE("different attrs is still equal") {
        positive_int embed_dim = 100_p;
        ComputationGraphOpAttrs other_attrs =
            ComputationGraphOpAttrs{MultiHeadAttentionAttrs{
                /*embed_dim=*/embed_dim,
                /*num_heads=*/num_heads,
                /*kdim=*/embed_dim,
                /*vdim=*/embed_dim,
                /*dropout=*/0.0,
                /*bias=*/true,
                /*add_bias_kv=*/false,
                /*add_zero_attn=*/false,
            }};

        LocalTaskRegistry task_registry = construct_local_task_registry_for_layers(
            {{layer_guid, LayerAttrs{attrs, std::nullopt}}});
        LocalTaskRegistry other_task_registry = construct_local_task_registry_for_layers(
            {{layer_guid, LayerAttrs{other_attrs, std::nullopt}}});

        CHECK(task_registry == other_task_registry);
      }

      SUBCASE("different layer_guid is not equal") {
        LocalTaskRegistry task_registry = construct_local_task_registry_for_layers(
            {{layer_guid, LayerAttrs{attrs, std::nullopt}}});
        layer_guid_t other_layer_guid = layer_guid_t{Node{1}};
        LocalTaskRegistry other_task_registry = construct_local_task_registry_for_layers(
            {{other_layer_guid, LayerAttrs{attrs, std::nullopt}}});

        CHECK(task_registry != other_task_registry);
      }
    }

    SUBCASE("registry_contains_task_for_layer") {
      SUBCASE("Task exists") {
        LocalTaskRegistry task_registry = construct_local_task_registry_for_layers({
            {layer_guid, LayerAttrs{attrs, std::nullopt}},
        });
        
        SUBCASE("Init") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::INIT);
          CHECK(result == true);
        }

        SUBCASE("Fwd") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::FWD);
          CHECK(result == true);
        }

        SUBCASE("Bwd") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::BWD);
          CHECK(result == true);
        }
      }

      SUBCASE("Partial task does not exist") {
        ComputationGraphOpAttrs bmm_attrs = ComputationGraphOpAttrs{
            BatchMatmulAttrs{/*a_seq_length_dim=*/10_n,
                             /*b_seq_length_dim=*/20_n}};
        LocalTaskRegistry task_registry = construct_local_task_registry_for_layers({
            {layer_guid, LayerAttrs{bmm_attrs, std::nullopt}},
        });

        SUBCASE("Init") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::INIT);
          CHECK(result == false);
        }

        SUBCASE("Fwd") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::FWD);
          CHECK(result == true);
        }

        SUBCASE("Bwd") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::BWD);
          CHECK(result == true);
        }
      }

      SUBCASE("Empty tasks") {
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            empty_task_ids = {{layer_guid, std::nullopt}};
        LocalTaskRegistry task_registry =
            LocalTaskRegistry{empty_task_ids, empty_task_ids, empty_task_ids, {}};

        SUBCASE("Init") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::INIT);
          CHECK(result == false);
        }

        SUBCASE("Fwd") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::FWD);
          CHECK(result == false);
        }

        SUBCASE("Bwd") {
          bool result = registry_contains_task_for_layer(
              task_registry, layer_guid, OpTaskType::BWD);
          CHECK(result == false);
        }
      }
    }
  }
}
