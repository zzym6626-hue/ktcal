#include "rknn_backend.h"
#include "rknn_api.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

static bool bad_shape = false;
static bool bad_layout = false;
static bool fail_release = false;
static int release_calls = 0;
static float first_input = 0;
static float x_logits[80], y_logits[60];

extern "C" {
int rknn_init(rknn_context* ctx, void*, uint32_t, uint32_t, rknn_init_extend*) {
  *ctx = 1; return 0;
}
int rknn_destroy(rknn_context) { return 0; }
int rknn_query(rknn_context, rknn_query_cmd cmd, void* info, uint32_t) {
  if (cmd == RKNN_QUERY_IN_OUT_NUM) {
    auto* count = static_cast<rknn_input_output_num*>(info);
    count->n_input = 1; count->n_output = 2;
  } else {
    auto* attr = static_cast<rknn_tensor_attr*>(info);
    if (cmd == RKNN_QUERY_INPUT_ATTR) {
      attr->n_elems = bad_shape ? 17 : 16*2*60*80;
      attr->fmt = bad_layout ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW;
      attr->n_dims = 5;
      attr->dims[0] = 1; attr->dims[1] = 16; attr->dims[2] = 2;
      attr->dims[3] = 60; attr->dims[4] = 80;
    } else { attr->n_elems = attr->index == 0 ? 80 : 60; }
  }
  return 0;
}
int rknn_inputs_set(rknn_context, uint32_t, rknn_input inputs[]) {
  assert(inputs[0].type == RKNN_TENSOR_FLOAT32);
  first_input = static_cast<float*>(inputs[0].buf)[0]; return 0;
}
int rknn_run(rknn_context, rknn_run_extend*) { return 0; }
int rknn_outputs_get(rknn_context, uint32_t, rknn_output outputs[], rknn_output_extend*) {
  outputs[0].buf = x_logits; outputs[0].size = sizeof(x_logits);
  outputs[1].buf = y_logits; outputs[1].size = sizeof(y_logits);
  return 0;
}
int rknn_outputs_release(rknn_context, uint32_t, rknn_output[]) {
  ++release_calls;
  return fail_release ? -1 : 0;
}
}

#ifndef KT_TEST_NO_MAIN
int main() {
  const std::string path = "/tmp/ktcal_mock_model_" + std::to_string(getpid()) + ".rknn";
  { std::ofstream model(path, std::ios::binary); model << "FAKE"; }
  x_logits[42] = 4; y_logits[19] = 7;
  ktcal::Window frames{};
  for (auto& frame : frames) frame.fill(30);
  {
    ktcal::RknnBackend backend(path.c_str(), 15.f);
    auto output = backend.run(frames);
    assert(output.x == 42 && output.y == 19);
    assert(std::abs(first_input - 2.f) < 1e-6f);
    assert(backend.start(frames).get().x == 42);
    fail_release = true;
    const int previous = release_calls;
    bool rejected_release = false;
    try { backend.run(frames); }
    catch (const std::runtime_error&) { rejected_release = true; }
    assert(rejected_release && release_calls == previous + 1);
    fail_release = false;
  }
  bad_shape = true;
  bool rejected = false;
  try { ktcal::RknnBackend backend(path.c_str(), 1.f); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  bad_shape = false;
  bad_layout = true;
  bool rejected_layout = false;
  try { ktcal::RknnBackend backend(path.c_str(), 1.f); }
  catch (const std::runtime_error&) { rejected_layout = true; }
  assert(rejected_layout);
  std::remove(path.c_str());
}
#endif
