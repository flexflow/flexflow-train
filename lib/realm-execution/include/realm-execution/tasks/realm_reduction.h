#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDUCTION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDUCTION_H
#include "op-attrs/datatype.dtg.h"
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
  static void apply(LHS &lhs, RHS rhs) {
    if (EXCLUSIVE) {
      lhs += rhs;
    } else {
      // Atomic float add via CAS loop
      union {
        float f;
        int i;
      } old_val, new_val;
      do {
        old_val.f = lhs;
        new_val.f = old_val.f + rhs;
      } while (
          !__sync_bool_compare_and_swap((int *)&lhs, old_val.i, new_val.i));
    }
  }

  /**
   * \brief Fold two RHS values: rhs1 += rhs2
   * \tparam EXCLUSIVE If true, direct addition; if false, atomic CAS loop
   * \param rhs1 Accumulator (modified in place)
   * \param rhs2 Value to fold in
   */
  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE) {
      rhs1 += rhs2;
    } else {
      // Atomic float add via CAS loop
      union {
        float f;
        int i;
      } old_val, new_val;
      do {
        old_val.f = rhs1;
        new_val.f = old_val.f + rhs2;
      } while (
          !__sync_bool_compare_and_swap((int *)&rhs1, old_val.i, new_val.i));
    }
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
  static void apply(LHS &lhs, RHS rhs) {
    if (EXCLUSIVE) {
      lhs += rhs;
    } else {
      // Atomic double add via CAS loop using long long reinterpretation
      union {
        double d;
        long long i;
      } old_val, new_val;
      do {
        old_val.d = lhs;
        new_val.d = old_val.d + rhs;
      } while (!__sync_bool_compare_and_swap(
          (long long *)&lhs, old_val.i, new_val.i));
    }
  }

  /**
   * \brief Fold two RHS values: rhs1 += rhs2
   * \tparam EXCLUSIVE If true, direct addition; if false, atomic CAS loop
   * \param rhs1 Accumulator (modified in place)
   * \param rhs2 Value to fold in
   */
  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE) {
      rhs1 += rhs2;
    } else {
      // Atomic double add via CAS loop using long long reinterpretation
      union {
        double d;
        long long i;
      } old_val, new_val;
      do {
        old_val.d = rhs1;
        new_val.d = old_val.d + rhs2;
      } while (!__sync_bool_compare_and_swap(
          (long long *)&rhs1, old_val.i, new_val.i));
    }
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
      PANIC("no sum reduction registered for datatype {}", dtype);
  }
}
} // namespace FlexFlow
#endif
