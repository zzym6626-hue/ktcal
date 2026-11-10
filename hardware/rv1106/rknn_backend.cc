#include "rknn_backend.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ktcal {
namespace {
void check(int status, const char* operation) {
  if (status != 0) throw std::runtime_error(std::string(operation) + " failed: " +
                                            std::to_string(status));
}
float decode(const rknn_output& output, std::size_t length) {
  if (!output.buf || output.size < length * sizeof(float))
    throw std::runtime_error("invalid RKNN logits output");
  const float* logits = static_cast<const float*>(output.buf);
  for (std::size_t i = 0; i < length; ++i)
    if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite RKNN output");
  return static_cast<float>(std::max_element(logits, logits + length) - logits);
}
} // namespace

RknnBackend::RknnBackend(const char* model_path, float input_divisor)
    : divisor_(input_divisor) {
  if (!(std::isfinite(divisor_) && divisor_ > 0))
    throw std::invalid_argument("input divisor must be positive and finite");
  std::ifstream file(model_path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open RKNN model");
  std::vector<char> model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (model.empty() || model.size() > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("invalid RKNN model size");
  check(rknn_init(&context_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr),
        "rknn_init");
  try {
    rknn_input_output_num number{};
    check(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &number, sizeof(number)), "query IO count");
    if (number.n_input != 1 || number.n_output != 2)
      throw std::runtime_error("RKNN model must have 1 input and 2 outputs");
    rknn_tensor_attr input{};
    input.index = 0;
    check(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)), "query input");
    const uint32_t dims[] = {1, kHistory, 2, kHeight, kWidth};
    if (input.fmt != RKNN_TENSOR_NCHW || input.n_dims != 5 ||
        input.n_elems != kHistory * 2 * kHeight * kWidth ||
        !std::equal(dims, dims + 5, input.dims))
      throw std::runtime_error("RKNN input must be NCHW [1,16,2,60,80]");
    for (uint32_t i = 0; i < 2; ++i) {
      rknn_tensor_attr output{};
      output.index = i;
      check(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)),
            "query output");
      if (output.n_elems != (i == 0 ? kWidth : kHeight))
        throw std::runtime_error("RKNN outputs must be ordered x[80], y[60] logits");
    }
  } catch (...) {
    rknn_destroy(context_);
    context_ = 0;
    throw;
  }
}

RknnBackend::~RknnBackend() {
  if (context_) rknn_destroy(context_);
}

Point RknnBackend::run(const Window& frames) {
  std::lock_guard<std::mutex> guard(mutex_); // serializes async and sync on one NPU context
  std::vector<float> input_data(kHistory * 2 * kWidth * kHeight);
  std::size_t offset = 0;
  for (const auto& frame : frames) for (float v : frame) {
    if (!std::isfinite(v)) throw std::invalid_argument("non-finite event frame");
    input_data[offset++] = v / divisor_;
  }
  rknn_input input{};
  input.index = 0;
  input.buf = input_data.data();
  input.size = static_cast<uint32_t>(input_data.size() * sizeof(float));
  input.type = RKNN_TENSOR_FLOAT32;
  input.fmt = RKNN_TENSOR_NCHW;
  check(rknn_inputs_set(context_, 1, &input), "rknn_inputs_set");
  check(rknn_run(context_, nullptr), "rknn_run");
  rknn_output outputs[2]{};
  outputs[0].want_float = 1; outputs[1].want_float = 1;
  outputs[0].index = 0; outputs[1].index = 1;
  check(rknn_outputs_get(context_, 2, outputs, nullptr), "rknn_outputs_get");
  Point point;
  try {
    point = {decode(outputs[0], kWidth), decode(outputs[1], kHeight)};
  } catch (...) {
    rknn_outputs_release(context_, 2, outputs);
    throw;
  }
  check(rknn_outputs_release(context_, 2, outputs), "rknn_outputs_release");
  return point;
}

std::future<Point> RknnBackend::start(Window frames) {
  return std::async(std::launch::async, [this, frames = std::move(frames)] {
    return run(frames);
  });
}
} // namespace ktcal
