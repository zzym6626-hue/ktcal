// Test double NPU: validates host adapter contracts, not Rockchip conversion/performance.
#include "rknn_sequence_backend.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

static int fault = 0, destroys = 0, releases = 0;
static bool fail_release = false;
static float first_raw = 0;
static std::vector<float> x(100*80), y(100*60);

extern "C" {
int rknn_init(rknn_context* context, void*, uint32_t, uint32_t, rknn_init_extend*) { *context = 1; return 0; }
int rknn_destroy(rknn_context) { ++destroys; return 0; }
int rknn_query(rknn_context, rknn_query_cmd command, void* output, uint32_t) {
  if (command == RKNN_QUERY_IN_OUT_NUM) {
    auto count = static_cast<rknn_input_output_num*>(output);
    count->n_input = 1; count->n_output = fault == 1 ? 1 : 2;
  } else {
    auto attr = static_cast<rknn_tensor_attr*>(output);
    if (command == RKNN_QUERY_INPUT_ATTR) {
      attr->n_dims = 5; attr->fmt = fault == 3 ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW;
      attr->dims[0] = 1; attr->dims[1] = fault == 2 ? 16 : 100;
      attr->dims[2] = 2; attr->dims[3] = 60; attr->dims[4] = 80;
      attr->n_elems = attr->dims[1] * 2 * 60 * 80;
    } else {
      attr->n_dims = 3; attr->dims[0] = 1; attr->dims[1] = fault == 4 ? 1 : 100;
      attr->dims[2] = attr->index == 0 ? 80 : 60;
      attr->n_elems = attr->dims[1] * attr->dims[2];
    }
  }
  return 0;
}
int rknn_inputs_set(rknn_context, uint32_t count, rknn_input inputs[]) {
  assert(count == 1 && inputs[0].size == 100 * sizeof(ktcal::Frame));
  assert(inputs[0].type == RKNN_TENSOR_FLOAT32);
  first_raw = static_cast<float*>(inputs[0].buf)[0]; return 0;
}
int rknn_run(rknn_context, rknn_run_extend*) { return fault == 5 ? -1 : 0; }
int rknn_outputs_get(rknn_context, uint32_t count, rknn_output outputs[], rknn_output_extend*) {
  assert(count == 2 && outputs[0].want_float && outputs[1].want_float);
  outputs[0].buf = x.data(); outputs[0].size = static_cast<uint32_t>(x.size()*sizeof(float));
  outputs[1].buf = y.data(); outputs[1].size = static_cast<uint32_t>(y.size()*sizeof(float));
  if (fault == 6) outputs[1].size -= sizeof(float);
  return 0;
}
int rknn_outputs_release(rknn_context, uint32_t, rknn_output[]) { ++releases; return fail_release ? -1 : 0; }
}

int main() {
  const auto path = "/tmp/ktcal_mock_sequence_" + std::to_string(getpid()) + ".rknn";
  { std::ofstream output(path, std::ios::binary); output << "TEST-ONLY"; }
  ktcal::Record record(100);
  record[0][0] = 30;
  x[42] = 7; x[99*80+19] = 8; y[30] = 9; y[99*60+17] = 10;
  {
    ktcal::RknnSequenceBackend backend(path.c_str());
    const auto output = backend.run(record);
    assert(first_raw == 30); // Raw input is not divided twice.
    assert(output.x.size() == 8000 && output.y.size() == 6000);
    assert(output.x[42] == 7 && output.x[99*80+19] == 8 && output.y[99*60+17] == 10);
    const auto coords = ktcal::decode_sequence(output);
    assert(coords[0][0] > 41 && coords[99][0] < 20);
    for (int error : {5,6}) {
      fault = error; bool failed = false; const int old_releases = releases;
      try { backend.run(record); } catch (const std::runtime_error&) { failed = true; }
      assert(failed); if (error == 6) assert(releases == old_releases + 1);
    }
    fault = 0; x[1] = std::numeric_limits<float>::quiet_NaN();
    const int before = releases; bool failed = false;
    try { backend.run(record); } catch (const std::runtime_error&) { failed = true; }
    assert(failed && releases == before + 1); x[1] = 0;
    fail_release = true; failed = false;
    try { backend.run(record); } catch (const std::runtime_error&) { failed = true; }
    assert(failed); fail_release = false;
    record[0][0] = std::numeric_limits<float>::infinity(); failed = false;
    try { backend.run(record); } catch (const std::invalid_argument&) { failed = true; }
    assert(failed); record[0][0] = 30;
    record.resize(16); failed = false;
    try { backend.run(record); } catch (const std::invalid_argument&) { failed = true; }
    assert(failed); record.resize(100);
  }
  for (int error : {1,2,3,4}) {
    fault = error; bool failed = false; const int before = destroys;
    try { ktcal::RknnSequenceBackend backend(path.c_str()); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed && destroys == before + 1);
  }
  std::istringstream empty;
  assert(!ktcal::read_record(empty, record));
  std::istringstream partial(std::string(sizeof(ktcal::Frame), '\0'));
  bool failed = false;
  try { ktcal::read_record(partial, record); } catch (const std::runtime_error&) { failed = true; }
  assert(failed);
  std::istringstream complete(std::string(sizeof(ktcal::Frame)*100, '\0'));
  assert(ktcal::read_record(complete, record) && record.size() == 100);
  assert(!ktcal::read_record(complete, record));
  std::remove(path.c_str());
}
