#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDUCTION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDUCTION_H
#include "op-attrs/datatype.dtg.h"
#include <cassert>
#include <realm.h>
namespace FlexFlow {

/**
 * \brief Realm Sum Reduction for Float
 * \see https://legion.stanford.edu/tutorial/realm/reductions.html
 */
struct SumReductionFloat {
  using LHS = float;
  using RHS = float;

  /** \brief Identity element for addition (0.0) */
  static constexpr RHS identity = 0.0f;

  /**
   * \brief Apply reduction: lhs += rhs
   * \tparam EXCLUSIVE If true, direct addition; if false, atomic CAS loop
   * \param lhs Left-hand side accumulator (modified in place)
   * \param rhs Value to add
   */
  template <bool EXCLUSIVE>
  REALM_CUDA_HD static void apply(LHS &lhs, RHS rhs) {
    if (EXCLUSIVE) {
      lhs += rhs;
    } else {
#if defined(__CUDA_ARCH__)
      atomicAdd(&lhs, rhs);
#else
      union {
        float f;
        int i;
      } old_val, new_val;
      do {
        old_val.f = lhs;
        new_val.f = old_val.f + rhs;
      } while (
          !__sync_bool_compare_and_swap((int *)&lhs, old_val.i, new_val.i));
#endif
    }
  }

  template <bool EXCLUSIVE>
  __device__ static void apply_cuda(LHS &lhs, RHS rhs) {
    apply<EXCLUSIVE>(lhs, rhs);
  }

  /**
   * \brief Fold two RHS values: rhs1 += rhs2
   * \tparam EXCLUSIVE If true, direct addition; if false, atomic CAS loop
   * \param rhs1 Accumulator (modified in place)
   * \param rhs2 Value to fold in
   */
  template <bool EXCLUSIVE>
  REALM_CUDA_HD static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE) {
      rhs1 += rhs2;
    } else {
#if defined(__CUDA_ARCH__)
      atomicAdd(&rhs1, rhs2);
#else
      union {
        float f;
        int i;
      } old_val, new_val;
      do {
        old_val.f = rhs1;
        new_val.f = old_val.f + rhs2;
      } while (
          !__sync_bool_compare_and_swap((int *)&rhs1, old_val.i, new_val.i));
#endif
    }
  }
  template <bool EXCLUSIVE>
  __device__ static void fold_cuda(RHS &rhs1, RHS rhs2) {
    fold<EXCLUSIVE>(rhs1, rhs2);
  }
};

/**
 * \brief Realm Sum Reduction for Double
 * \see https://legion.stanford.edu/tutorial/realm/reductions.html
 */
struct SumReductionDouble {
  using LHS = double;
  using RHS = double;

  /** \brief Identity element for addition (0.0) */
  static constexpr RHS identity = 0.0;

  /**
   * \brief Apply reduction: lhs += rhs
   * \tparam EXCLUSIVE If true, direct addition; if false, atomic CAS loop
   * \param lhs Left-hand side accumulator (modified in place)
   * \param rhs Value to add
   */
  template <bool EXCLUSIVE>
  REALM_CUDA_HD static void apply(LHS &lhs, RHS rhs) {
    if (EXCLUSIVE) {
      lhs += rhs;
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      atomicAdd(&lhs, rhs);
#elif defined(__CUDA_ARCH__)
      // pre-Pascal fallback CAS loop
      unsigned long long int *addr = (unsigned long long int *)&lhs;
      unsigned long long int old = *addr, assumed;
      do {
        assumed = old;
        old = atomicCAS(
            addr,
            assumed,
            __double_as_longlong(rhs + __longlong_as_double(assumed)));
      } while (assumed != old);
#else
      union {
        double d;
        long long i;
      } old_val, new_val;
      do {
        old_val.d = lhs;
        new_val.d = old_val.d + rhs;
      } while (!__sync_bool_compare_and_swap(
          (long long *)&lhs, old_val.i, new_val.i));
#endif
    }
  }
  template <bool EXCLUSIVE>
  __device__ static void apply_cuda(LHS &lhs, RHS rhs) {
    apply<EXCLUSIVE>(lhs, rhs);
  }

  /**
   * \brief Fold two RHS values: rhs1 += rhs2
   * \tparam EXCLUSIVE If true, direct addition; if false, atomic CAS loop
   * \param rhs1 Accumulator (modified in place)
   * \param rhs2 Value to fold in
   */
  template <bool EXCLUSIVE>
  REALM_CUDA_HD static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE) {
      rhs1 += rhs2;
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      atomicAdd(&rhs1, rhs2);
#elif defined(__CUDA_ARCH__)
      unsigned long long int *addr = (unsigned long long int *)&rhs1;
      unsigned long long int old = *addr, assumed;
      do {
        assumed = old;
        old = atomicCAS(
            addr,
            assumed,
            __double_as_longlong(rhs2 + __longlong_as_double(assumed)));
      } while (assumed != old);
#else
      union {
        double d;
        long long i;
      } old_val, new_val;
      do {
        old_val.d = rhs1;
        new_val.d = old_val.d + rhs2;
      } while (!__sync_bool_compare_and_swap(
          (long long *)&rhs1, old_val.i, new_val.i));
#endif
    }
  }

  template <bool EXCLUSIVE>
  __device__ static void fold_cuda(RHS &rhs1, RHS rhs2) {
    fold<EXCLUSIVE>(rhs1, rhs2);
  }
};

/**
 * \brief Reduction op IDs for sum reductions
 * \warning These IDs must not conflict with other registered reduction ops
 */
enum SumReductionOpIDs {
  REDOP_SUM_FLOAT = 1,  ///< Sum reduction op ID for float
  REDOP_SUM_DOUBLE = 2, ///< Sum reduction op ID for double
};

/**
 * \brief Returns the Realm reduction op ID for a sum reduction over the given datatype
 * \param dtype The datatype to look up
 * \return The corresponding Realm::ReductionOpID
 * \throws PANIC if no sum reduction is registered for the given datatype
 */
inline Realm::ReductionOpID get_sum_reduction_op_id(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT:
      return REDOP_SUM_FLOAT;
    case DataType::DOUBLE:
      return REDOP_SUM_DOUBLE;
    default:
#ifndef __CUDA_ARCH__
      throw std::runtime_error("no sum reduction registered for datatype");
#else
      assert(false);
      return REDOP_SUM_FLOAT; //unreachable
#endif
  }
}
} // namespace FlexFlow
#endif
