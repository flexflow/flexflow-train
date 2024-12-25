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

#include "device.h"
#include "kernels/metrics_kernels.h"
#include "kernels/perf_metrics.h"
#include "pcg/metric.h"

namespace FlexFlow {

struct CUDAPerfMetrics {
  int train_all;
  int train_correct;
  float cce_loss;
  float sparse_cce_loss;
  float mse_loss;
  float rmse_loss;
  float mae_loss;
  double start_time;
  double current_time;

  CUDAPerfMetrics() = delete;
  CUDAPerfMetrics(PerfMetrics const &perf)
      : train_all(perf.train_all),
        train_correct(perf.train_correct.value_or(-1)),
        cce_loss(perf.cce_loss.value_or(-1)),
        sparse_cce_loss(perf.sparse_cce_loss.value_or(-1)),
        mse_loss(perf.mse_loss.value_or(-1)),
        rmse_loss(perf.rmse_loss.value_or(-1)),
        mae_loss(perf.mae_loss.value_or(-1)), start_time(perf.start_time),
        current_time(perf.current_time) {}
};

float const LOG_MIN_VALUE = 0.00000001f;

__global__ void update_metrics_sparse_label_kernel(float const *logits,
                                                   int const *labels,
                                                   CUDAPerfMetrics *perf,
                                                   const MetricsAttrs metrics,
                                                   int num_samples,
                                                   int num_classes) {
  CUDA_KERNEL_LOOP(b, num_samples) {
    if (metrics.measure_accuracy) {
      float max_val = -1.0f;
      int my_label = -1;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b * num_classes + i];
        if (my_logit > max_val) {
          max_val = my_logit;
          my_label = i;
        }
      }
      assert(my_label >= 0);
      atomicAdd(&(perf->train_all), 1);
      if (labels[b] == my_label) {
        atomicAdd(&(perf->train_correct), 1);
      }
    }
    if (metrics.measure_sparse_categorical_crossentropy) {
      float my_logit = max(logits[b * num_classes + labels[b]], LOG_MIN_VALUE);
      atomicAdd(&(perf->sparse_cce_loss), -log(my_logit));
    }
    if (metrics.measure_mean_squared_error ||
        metrics.measure_root_mean_squared_error ||
        metrics.measure_mean_absolute_error) {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b * num_classes + i];
        float my_label = (labels[b] == i) ? 1.0f : 0.0f;
        mse += (my_logit - my_label) * (my_logit - my_label);
        mae += abs(my_logit - my_label);
      }
      if (metrics.measure_mean_squared_error) {
        atomicAdd(&(perf->mse_loss), mse);
      }
      if (metrics.measure_root_mean_squared_error) {
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      }
      if (metrics.measure_mean_absolute_error) {
        atomicAdd(&(perf->mae_loss), mae);
      }
    }
  }
}

__global__ void update_metrics_label_kernel(float const *logits,
                                            float const *labels,
                                            CUDAPerfMetrics *perf,
                                            const MetricsAttrs metrics,
                                            int num_samples,
                                            int num_classes) {
  CUDA_KERNEL_LOOP(b, num_samples) {
    atomicAdd(&(perf->train_all), 1);
    if (metrics.measure_accuracy) {
      if (num_classes == 1) {
        // accuracy does not make sense when num_classes = 1
        // we just return 100%
        atomicAdd(&(perf->train_all), 1);
        atomicAdd(&(perf->train_correct), 1);
      } else {
        float max_val = 0.0f;
        int my_label = -1, true_label = -1;
        for (int i = 0; i < num_classes; i++) {
          if (my_label == -1 || logits[b * num_classes + i] > max_val) {
            max_val = logits[b * num_classes + i];
            my_label = i;
          }
          if (labels[b * num_classes + i] > 0.9f) {
            assert(true_label == -1);
            true_label = i;
          }
        }
        assert(my_label >= 0);
        assert(true_label >= 0);
        if (true_label == my_label) {
          atomicAdd(&(perf->train_correct), 1);
        }
      }
    }
    if (metrics.measure_categorical_crossentropy) {
      float cce = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        if (labels[b * num_classes + i] > 0.0f) {
          float my_logit = max(logits[b * num_classes + i], LOG_MIN_VALUE);
          cce += labels[b * num_classes + i] * -log(my_logit);
        }
      }
      atomicAdd(&(perf->cce_loss), cce);
    }
    if (metrics.measure_mean_squared_error ||
        metrics.measure_root_mean_squared_error ||
        metrics.measure_mean_absolute_error) {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float diff = logits[b * num_classes + i] - labels[b * num_classes + i];
        mse += diff * diff;
        mae += abs(diff);
      }
      if (metrics.measure_mean_squared_error) {
        atomicAdd(&(perf->mse_loss), mse);
      }
      if (metrics.measure_root_mean_squared_error) {
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      }
      if (metrics.measure_mean_absolute_error) {
        atomicAdd(&(perf->mae_loss), mae);
      }
    }
  }
}

void update_metrics_sparse_label_kernel_wrapper(float const *logit_ptr,
                                                int const *label_ptr,
                                                MetricsAttrs const *me,
                                                int num_effective_samples,
                                                int num_classes,
                                                PerfMetrics &perf_zc) {
  CUDAPerfMetrics perf(perf_zc);
  CUDAPerfMetrics *perf_cuda;
  checkCUDA(cudaMalloc(&perf_cuda, sizeof(CUDAPerfMetrics)));
  checkCUDA(cudaMemcpy(
      perf_cuda, &perf, sizeof(CUDAPerfMetrics), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  update_metrics_sparse_label_kernel<<<GET_BLOCKS(num_effective_samples),
                                       CUDA_NUM_THREADS,
                                       0,
                                       stream>>>(
      logit_ptr, label_ptr, perf_cuda, *me, num_effective_samples, num_classes);
  checkCUDA(cudaStreamSynchronize(stream));
  checkCUDA(cudaMemcpy(
      &perf, perf_cuda, sizeof(CUDAPerfMetrics), cudaMemcpyDeviceToHost));
  checkCUDA(cudaFree(perf_cuda));
}

void update_metrics_label_kernel_wrapper(float const *logit_ptr,
                                         float const *label_ptr,
                                         MetricsAttrs const *me,
                                         int num_samples,
                                         int num_classes,
                                         PerfMetrics &perf_zc) {
  CUDAPerfMetrics perf(perf_zc);
  CUDAPerfMetrics *perf_cuda;
  checkCUDA(cudaMalloc(&perf_cuda, sizeof(CUDAPerfMetrics)));
  checkCUDA(cudaMemcpy(
      perf_cuda, &perf, sizeof(CUDAPerfMetrics), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  update_metrics_label_kernel<<<GET_BLOCKS(num_samples), 256, 0, stream>>>(
      logit_ptr, label_ptr, perf_cuda, *me, num_samples, num_classes);
  checkCUDA(cudaStreamSynchronize(stream));
  checkCUDA(cudaMemcpy(
      &perf, perf_cuda, sizeof(CUDAPerfMetrics), cudaMemcpyDeviceToHost));
  checkCUDA(cudaFree(perf_cuda));
}

}; // namespace FlexFlow
