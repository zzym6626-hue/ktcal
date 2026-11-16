#include "rknn_sequence_backend.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace ktcal {
namespace {
void check(int status, const char* operation) {
  if (status != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
void validate_output(const rknn_output& output, std::size_t elements) {
  if (!output.buf || output.size != elements * sizeof(float))
    throw std::runtime_error("invalid sequence RKNN output size");
  const float* values = static_cast<const float*>(output.buf);
  for (std::size_t i = 0; i < elements; ++i)
    if (!std::isfinite(values[i])) throw std::runtime_error("non-finite RKNN logits");
}
}
RknnSequenceBackend::RknnSequenceBackend(const char* model_path) {
  std::ifstream input(model_path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open RKNN model");
  std::vector<char> model((std::istreambuf_iterator<char>(input)), {});
  if (model.empty() || model.size() > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("invalid RKNN model size");
  check(rknn_init(&context_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr), "rknn_init");
  try {
    rknn_input_output_num count{};
    check(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "query count");
    if (count.n_input != 1 || count.n_output != 2) throw std::runtime_error("expected one input and two logits outputs");
    rknn_tensor_attr attr{};
    check(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "query input");
    const uint32_t dims[5] = {1,100,2,60,80};
    if (attr.n_dims != 5 || attr.fmt != RKNN_TENSOR_NCHW ||
        attr.n_elems != 100 * 2 * 60 * 80 || !std::equal(dims, dims + 5, attr.dims))
      throw std::runtime_error("expected NCHW [1,100,2,60,80] RKNN input; converted model contract differs");
    for (uint32_t i = 0; i < 2; ++i) {
      attr = {}; attr.index = i;
      check(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "query output");
      const uint32_t bins = i == 0 ? kWidth : kHeight;
      if (attr.n_dims != 3 || attr.dims[0] != 1 || attr.dims[1] != 100 ||
          attr.dims[2] != bins || attr.n_elems != 100 * bins)
        throw std::runtime_error("expected ordered x[1,100,80],y[1,100,60] RKNN outputs");
    }
  } catch (...) { rknn_destroy(context_); context_ = 0; throw; }
}
RknnSequenceBackend::~RknnSequenceBackend() {
  if (context_) rknn_destroy(context_);
}
SequenceLogits RknnSequenceBackend::run(const Record& frames) {
  if (frames.size() != kSequence) throw std::invalid_argument("RKNN requires exactly 100 frames");
  for (const auto& frame : frames) for (float value : frame)
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite raw frame");
  // Record is contiguous planar raw float32. RKNN copies/converts synchronously.
  rknn_input input{};
  input.buf = const_cast<Frame*>(frames.data());
  input.size = static_cast<uint32_t>(frames.size() * sizeof(Frame));
  input.type = RKNN_TENSOR_FLOAT32; input.fmt = RKNN_TENSOR_NCHW;
  check(rknn_inputs_set(context_, 1, &input), "rknn_inputs_set");
  check(rknn_run(context_, nullptr), "rknn_run");
  rknn_output outputs[2]{};
  outputs[0].index = 0; outputs[1].index = 1;
  outputs[0].want_float = outputs[1].want_float = 1;
  check(rknn_outputs_get(context_, 2, outputs, nullptr), "rknn_outputs_get");
  SequenceLogits result;
  try {
    validate_output(outputs[0], 100 * kWidth); validate_output(outputs[1], 100 * kHeight);
    const auto x = static_cast<const float*>(outputs[0].buf), y = static_cast<const float*>(outputs[1].buf);
    result.x.assign(x, x + 100 * kWidth); result.y.assign(y, y + 100 * kHeight);
  } catch (...) { rknn_outputs_release(context_, 2, outputs); throw; }
  check(rknn_outputs_release(context_, 2, outputs), "rknn_outputs_release");
  return result;
}
} // namespace ktcal
