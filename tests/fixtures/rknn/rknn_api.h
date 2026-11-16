// Minimal TEST DOUBLE declarations for host ABI-call tests. Not a vendor SDK.
// Production builds must include the matching official Rockchip rknn_api.h.
#pragma once
#include <cstdint>
extern "C" {
typedef uint64_t rknn_context;
enum rknn_query_cmd { RKNN_QUERY_IN_OUT_NUM, RKNN_QUERY_INPUT_ATTR, RKNN_QUERY_OUTPUT_ATTR };
enum rknn_tensor_format { RKNN_TENSOR_NCHW, RKNN_TENSOR_NHWC };
enum rknn_tensor_type { RKNN_TENSOR_FLOAT32 };
struct rknn_init_extend;
struct rknn_run_extend;
struct rknn_output_extend;
struct rknn_input_output_num { uint32_t n_input, n_output; };
struct rknn_tensor_attr {
  uint32_t index, n_dims, dims[16], n_elems;
  rknn_tensor_format fmt;
};
struct rknn_input {
  uint32_t index; void* buf; uint32_t size; uint8_t pass_through;
  rknn_tensor_type type; rknn_tensor_format fmt;
};
struct rknn_output { uint8_t want_float, is_prealloc; uint32_t index; void* buf; uint32_t size; };
int rknn_init(rknn_context*, void*, uint32_t, uint32_t, rknn_init_extend*);
int rknn_destroy(rknn_context);
int rknn_query(rknn_context, rknn_query_cmd, void*, uint32_t);
int rknn_inputs_set(rknn_context, uint32_t, rknn_input[]);
int rknn_run(rknn_context, rknn_run_extend*);
int rknn_outputs_get(rknn_context, uint32_t, rknn_output[], rknn_output_extend*);
int rknn_outputs_release(rknn_context, uint32_t, rknn_output[]);
}
