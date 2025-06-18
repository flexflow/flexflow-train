#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_METRICS_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_METRICS_KERNELS_H

#include "kernels/perf_metrics.h"
#include "pcg/metric_attrs.h"

namespace FlexFlow {

void update_metrics_sparse_label_kernel_wrapper(float const *logit_ptr,
                                                int const *label_ptr,
                                                MetricsAttrs const &me,
                                                int num_effective_samples,
                                                int num_classes,
                                                PerfMetrics &perf_zc);

void update_metrics_label_kernel_wrapper(float const *logit_ptr,
                                         float const *label_ptr,
                                         MetricsAttrs const &me,
                                         int num_samples,
                                         int num_classes,
                                         PerfMetrics &perf_zc);
} // namespace FlexFlow

#endif
