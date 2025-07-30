#include "local-execution/local_task_registry.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/local_cost_estimator.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/operator_task_set.h"
#include "local-execution/registered_task.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/task_signature_impl.h"
#include "utils/fmt/optional.h"
#include "utils/fmt/unordered_map.h"
#include <doctest/doctest.h>

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

    OperatorTaskSet mha_task_set = get_task_set_for_operator(attrs);
    {
      OperatorTaskSet expected_mha_task_set = OperatorTaskSet{
          /*init_task=*/registered_task_t{task_id_t::ATTENTION_INIT_TASK_ID},
          /*fwd_task=*/registered_task_t{task_id_t::ATTENTION_FWD_TASK_ID},
          /*bwd_task=*/registered_task_t{task_id_t::ATTENTION_BWD_TASK_ID},
      };
      REQUIRE(mha_task_set == expected_mha_task_set);
    }

    std::unordered_map<task_id_t, TaskSignatureAndImpl> mha_task_mapping = {
        {task_id_t::ATTENTION_INIT_TASK_ID,
         get_task_signature_and_impl_for_task_id(
             task_id_t::ATTENTION_INIT_TASK_ID)},
        {task_id_t::ATTENTION_FWD_TASK_ID,
         get_task_signature_and_impl_for_task_id(
             task_id_t::ATTENTION_FWD_TASK_ID)},
        {task_id_t::ATTENTION_BWD_TASK_ID,
         get_task_signature_and_impl_for_task_id(
             task_id_t::ATTENTION_BWD_TASK_ID)},
    };

    SUBCASE("register single layer") {
      LocalTaskRegistry task_registry =
          construct_local_task_registry_for_layers(
              {{layer_guid, LayerAttrs{attrs, std::nullopt}}});

      LocalTaskRegistry correct_task_registry = [&] {
        std::unordered_map<layer_guid_t, OperatorTaskSet> task_sets = {
            {
                layer_guid,
                mha_task_set,
            },
        };

        return LocalTaskRegistry{
            /*task_sets=*/{
                {layer_guid, mha_task_set},
            },
            /*task_mapping=*/mha_task_mapping,
        };
      }();

      CHECK(task_registry == correct_task_registry);
    }

    SUBCASE("multiple layers same task") {
      layer_guid_t other_layer_guid = layer_guid_t{Node{1}};
      LocalTaskRegistry task_registry =
          construct_local_task_registry_for_layers({
              {layer_guid, LayerAttrs{attrs, std::nullopt}},
              {other_layer_guid, LayerAttrs{attrs, std::nullopt}},
          });

      SUBCASE("layer to task ids") {
        std::unordered_map<layer_guid_t, OperatorTaskSet> correct = {
            {layer_guid, mha_task_set},
            {other_layer_guid, mha_task_set},
        };
        CHECK(task_registry.task_sets == correct);
      }

      SUBCASE("task to signature+impl mapping") {
        std::unordered_map<task_id_t, TaskSignatureAndImpl> correct =
            mha_task_mapping;

        CHECK(task_registry.task_mapping == correct);
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
      LocalTaskRegistry task_registry =
          construct_local_task_registry_for_layers({
              {layer_guid, LayerAttrs{attrs, std::nullopt}},
              {layer_1, LayerAttrs{attrs, std::nullopt}},
              {layer_2, LayerAttrs{other_attrs, std::nullopt}},
          });

      std::unordered_map<task_id_t, TaskSignatureAndImpl> correct_task_mapping =
          mha_task_mapping;

      CHECK(task_registry.task_mapping == correct_task_mapping);
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

        LocalTaskRegistry task_registry =
            construct_local_task_registry_for_layers(
                {{layer_guid, LayerAttrs{attrs, std::nullopt}}});
        LocalTaskRegistry other_task_registry =
            construct_local_task_registry_for_layers(
                {{layer_guid, LayerAttrs{other_attrs, std::nullopt}}});

        CHECK(task_registry == other_task_registry);
      }

      SUBCASE("different layer_guid is not equal") {
        LocalTaskRegistry task_registry =
            construct_local_task_registry_for_layers(
                {{layer_guid, LayerAttrs{attrs, std::nullopt}}});
        layer_guid_t other_layer_guid = layer_guid_t{Node{1}};
        LocalTaskRegistry other_task_registry =
            construct_local_task_registry_for_layers(
                {{other_layer_guid, LayerAttrs{attrs, std::nullopt}}});

        CHECK(task_registry != other_task_registry);
      }
    }

    SUBCASE("try_get_registered_task") {
      SUBCASE("Task exists") {
        LocalTaskRegistry task_registry =
            construct_local_task_registry_for_layers({
                {layer_guid, LayerAttrs{attrs, std::nullopt}},
            });

        SUBCASE("Init") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::INIT);
          std::optional<registered_task_t> correct = registered_task_t{
              task_id_t::ATTENTION_INIT_TASK_ID,
          };

          CHECK(result == correct);
        }

        SUBCASE("Fwd") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::FWD);
          std::optional<registered_task_t> correct = registered_task_t{
              task_id_t::ATTENTION_FWD_TASK_ID,
          };

          CHECK(result == correct);
        }

        SUBCASE("Bwd") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::BWD);
          std::optional<registered_task_t> correct = registered_task_t{
              task_id_t::ATTENTION_BWD_TASK_ID,
          };

          CHECK(result == correct);
        }
      }

      SUBCASE("Partial task does not exist") {
        ComputationGraphOpAttrs bmm_attrs = ComputationGraphOpAttrs{
            BatchMatmulAttrs{
                /*a_seq_length_dim=*/10_p,
                /*b_seq_length_dim=*/20_p,
            },
        };

        LocalTaskRegistry task_registry =
            construct_local_task_registry_for_layers({
                {layer_guid, LayerAttrs{bmm_attrs, std::nullopt}},
            });

        SUBCASE("Init") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::INIT);
          std::optional<registered_task_t> correct =
              make_noop_registered_task();

          CHECK(result == correct);
        }

        SUBCASE("Fwd") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::FWD);
          std::optional<registered_task_t> correct = registered_task_t{
              task_id_t::BATCHMATMUL_FWD_TASK_ID,
          };

          CHECK(result == correct);
        }

        SUBCASE("Bwd") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::BWD);
          std::optional<registered_task_t> correct = registered_task_t{
              task_id_t::BATCHMATMUL_BWD_TASK_ID,
          };

          CHECK(result == correct);
        }
      }

      SUBCASE("Empty tasks") {
        LocalTaskRegistry task_registry = LocalTaskRegistry{
            /*task_sets=*/{},
            /*task_mapping=*/{},
        };

        SUBCASE("Init") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::INIT);
          std::optional<registered_task_t> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("Fwd") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::FWD);
          std::optional<registered_task_t> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("Bwd") {
          std::optional<registered_task_t> result = try_get_registered_task(
              task_registry, layer_guid, OpTaskType::BWD);
          std::optional<registered_task_t> correct = std::nullopt;

          CHECK(result == correct);
        }
      }
    }
  }
}
