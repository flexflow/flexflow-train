#pragma once
#include <realm.h>
#include "op-attrs/datatype.dtg.h"

namespace FlexFlow {

// Sum reduction for float
struct SumReductionFloat {
  using LHS = float;
  using RHS = float;
  static constexpr RHS identity = 0.0f;  // ← inside struct, constexpr

  template <bool EXCLUSIVE>
  static void apply(LHS &lhs, RHS rhs) {
    if (EXCLUSIVE) {
      lhs += rhs;
    } else {
      // atomic add for non-exclusive
      __sync_fetch_and_add((int*)&lhs, *(int*)&rhs);  
      // proper float atomic add — use union trick
      union { float f; int i; } old_val, new_val;
      do {
        old_val.f = lhs;
        new_val.f = old_val.f + rhs;
      } while (!__sync_bool_compare_and_swap(
          (int*)&lhs, old_val.i, new_val.i));
    }
  }

  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE) {
      rhs1 += rhs2;
    } else {
      union { float f; int i; } old_val, new_val;
      do {
        old_val.f = rhs1;
        new_val.f = old_val.f + rhs2;
      } while (!__sync_bool_compare_and_swap(
          (int*)&rhs1, old_val.i, new_val.i));
    }
  }
};


// Sum reduction for double
struct SumReductionDouble {
  using LHS = double;
  using RHS = double;
  static constexpr RHS identity = 0.0;  // ← inside struct, constexpr  

  template <bool EXCLUSIVE>
  static void apply(LHS &lhs, RHS rhs) {
    if (EXCLUSIVE) {
      lhs += rhs;
    } else {
      union { double d; long long i; } old_val, new_val;
      do {
        old_val.d = lhs;
        new_val.d = old_val.d + rhs;
      } while (!__sync_bool_compare_and_swap(
          (long long*)&lhs, old_val.i, new_val.i));
    }
  }

  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE) {
      rhs1 += rhs2;
    } else {
      union { double d; long long i; } old_val, new_val;
      do {
        old_val.d = rhs1;
        new_val.d = old_val.d + rhs2;
      } while (!__sync_bool_compare_and_swap(
          (long long*)&rhs1, old_val.i, new_val.i));
    }
  }
};

// Reduction op IDs — must not conflict with other registered redops
enum SumReductionOpIDs {
  REDOP_SUM_FLOAT  = 1,
  REDOP_SUM_DOUBLE = 2,
};

inline Realm::ReductionOpID get_sum_reduction_op_id(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT:  return REDOP_SUM_FLOAT;
    case DataType::DOUBLE: return REDOP_SUM_DOUBLE;
    default:
      PANIC("no sum reduction registered for datatype {}", dtype);
  }
}

} // namespace FlexFlow
