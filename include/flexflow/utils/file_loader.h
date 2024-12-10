/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"

using namespace std;
using namespace FlexFlow;
using namespace Legion;

class FileDataLoader {
public:
  FileDataLoader(std::string _prompts_filepath,
                 std::string _weights_folder,
                 int _num_heads,
                 int _num_kv_heads,
                 size_t _hidden_dim,
                 size_t _qkv_inner_dim,
                 int _tensor_parallelism_degree,
                 bool _use_full_precision);

  BatchConfig::TokenId *generate_requests(int num, int length);

  template <typename DT>
  void load_single_weight_tensor(FFModel *ff,
                                 Layer *l,
                                 int weight_idx,
                                 size_t volume,
                                 size_t num_replicas,
                                 DT *weight,
                                 Domain weight_domain);

  void load_quantization_weight(FFModel *ff,
                                Layer *l,
                                int weight_idx,
                                size_t volume,
                                size_t num_replicas,
                                char *weight,
                                DataType data_type,
                                Domain weight_domain);

  static void
      load_weight_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  void load_weights_parallel(FFModel *ff,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

  void load_positions(FFModel *ff,
                      Tensor pt,
                      ParallelTensor position_pt,
                      int max_seq_length,
                      int offset);

private:
  int num_heads, num_kv_heads, tensor_parallelism_degree;
  size_t hidden_dim, qkv_inner_dim;
  std::string prompts_filepath;
  std::string weights_folder;
  bool use_full_precision;
};

struct WeightLoadTaskArgs {
  FFModel *ff;
  FileDataLoader *loader;
  Layer *layer;
  int weight_idx;
  size_t volume, num_replicas;
  DataType data_type;
  WeightLoadTaskArgs(FFModel *_ff,
                     FileDataLoader *_loader,
                     Layer *_l,
                     int _idx,
                     size_t _volume,
                     size_t _num_replicas,
                     DataType _data_type)
      : ff(_ff), loader(_loader), layer(_l), weight_idx(_idx), volume(_volume),
        num_replicas(_num_replicas), data_type(_data_type) {}
};
