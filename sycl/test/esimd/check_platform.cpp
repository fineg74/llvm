// RUN: not %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,intel_gpu_acm_g10 -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="error:"

// This test checks that target specific constraints are enforced for lsc API:
// - successfully compiles code that is valid for both DG2 and PVC
// - emit an error for code that is invalid for DG2

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl;

template <class T, int N>
SYCL_EXTERNAL auto test_load(T *ptr) SYCL_ESIMD_FUNCTION {
  return lsc_block_load<T, N>(ptr);
}

template <class T, int N>
SYCL_EXTERNAL auto atomic_test_load(T *ptr) SYCL_ESIMD_FUNCTION {
  return atomic_update<sycl::ext::intel::esimd::native::lsc::atomic_op::load, T,
                       N>(ptr, 0, 1);
}

// --- Positive tests.

template auto test_load<uint32_t, 64>(uint32_t *) SYCL_ESIMD_FUNCTION;
template auto atomic_test_load<uint32_t, 16>(uint32_t *) SYCL_ESIMD_FUNCTION;

// --- Negative tests.

template auto test_load<uint64_t, 64>(uint64_t *) SYCL_ESIMD_FUNCTION;
// CHECK: {{.*}}error: {{.*}} Unsupported architecture.
template auto atomic_test_load<int32_t, 32>(int32_t *) SYCL_ESIMD_FUNCTION;
// CHECK: {{.*}}error: {{.*}} Unsupported architecture.
