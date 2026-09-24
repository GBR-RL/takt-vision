#include "takt/infer/onnx_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace takt {
namespace {

// ONNX Runtime keeps its logging and global thread pools in the environment; one per process.
Ort::Env& shared_env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "takt");
  return env;
}

bool has_dynamic_dim(const std::vector<std::int64_t>& shape) {
  return std::any_of(shape.begin(), shape.end(), [](std::int64_t d) { return d <= 0; });
}

std::string shape_to_string(const std::vector<std::int64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + "]";
}

}  // namespace

struct OnnxBackend::Impl {
  Ort::SessionOptions session_options;
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory_info{nullptr};
  Ort::RunOptions run_options;
  TensorSpec input_spec;
  TensorSpec output_spec;
  std::size_t input_elements = 0;
  std::size_t output_elements = 0;
  std::string name;
};

OnnxBackend::OnnxBackend(const OnnxBackendOptions& options) : impl_(std::make_unique<Impl>()) {
  Impl& im = *impl_;
  Ort::SessionOptions& so = im.session_options;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  so.SetInterOpNumThreads(1);
  if (options.intra_op_threads > 0) so.SetIntraOpNumThreads(options.intra_op_threads);
  so.AddConfigEntry("session.intra_op.allow_spinning", options.allow_spinning ? "1" : "0");

  if (options.execution_provider == "cpu") {
    im.name = "onnxruntime-cpu";
  } else if (options.execution_provider == "cuda") {
    try {
      OrtCUDAProviderOptions cuda_options{};
      so.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception& e) {
      throw std::runtime_error(
          std::string("CUDA execution provider unavailable (use a GPU build of "
                      "ONNX Runtime): ") +
          e.what());
    }
    im.name = "onnxruntime-cuda";
  } else {
    throw std::invalid_argument("OnnxBackend: unknown execution provider '" +
                                options.execution_provider + "' (expected cpu or cuda)");
  }

  try {
    // path::c_str() yields ORTCHAR_T* on every platform: wchar_t on Windows, char elsewhere.
    im.session = Ort::Session(shared_env(), options.model_path.c_str(), so);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("OnnxBackend: failed to load '" + options.model_path.string() +
                             "': " + e.what());
  }
  im.memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  if (im.session.GetInputCount() != 1) {
    throw std::runtime_error("OnnxBackend: expected a model with exactly one input");
  }
  if (im.session.GetOutputCount() < 1)
    throw std::runtime_error("OnnxBackend: model has no outputs");

  Ort::AllocatorWithDefaultOptions allocator;
  im.input_spec.name = im.session.GetInputNameAllocated(0, allocator).get();
  im.output_spec.name = im.session.GetOutputNameAllocated(0, allocator).get();

  const Ort::TypeInfo input_type = im.session.GetInputTypeInfo(0);
  const auto input_tensor = input_type.GetTensorTypeAndShapeInfo();
  if (input_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(
        "OnnxBackend: only float32 inputs are supported (export with half=False)");
  }
  im.input_spec.shape = input_tensor.GetShape();
  if (im.input_spec.shape.size() != 4) {
    throw std::runtime_error("OnnxBackend: expected an NCHW input, got " +
                             shape_to_string(im.input_spec.shape));
  }
  // Resolve dynamic axes: batch 1, spatial size from the options.
  if (im.input_spec.shape[0] <= 0) im.input_spec.shape[0] = 1;
  if (im.input_spec.shape[1] <= 0) im.input_spec.shape[1] = 3;
  for (std::size_t axis = 2; axis < 4; ++axis) {
    if (im.input_spec.shape[axis] <= 0) im.input_spec.shape[axis] = options.default_input_size;
  }
  im.input_elements = im.input_spec.element_count();

  const Ort::TypeInfo output_type = im.session.GetOutputTypeInfo(0);
  const auto output_tensor = output_type.GetTensorTypeAndShapeInfo();
  if (output_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("OnnxBackend: only float32 outputs are supported");
  }
  im.output_spec.shape = output_tensor.GetShape();
  if (has_dynamic_dim(im.output_spec.shape)) {
    // Discover the concrete output shape with one dry run on a zero tensor.
    std::vector<float> zeros(im.input_elements, 0.0f);
    Ort::Value input =
        Ort::Value::CreateTensor<float>(im.memory_info, zeros.data(), zeros.size(),
                                        im.input_spec.shape.data(), im.input_spec.shape.size());
    const char* input_name = im.input_spec.name.c_str();
    const char* output_name = im.output_spec.name.c_str();
    std::vector<Ort::Value> outputs =
        im.session.Run(im.run_options, &input_name, &input, 1, &output_name, 1);
    im.output_spec.shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  }
  im.output_elements = im.output_spec.element_count();
}

OnnxBackend::~OnnxBackend() = default;
OnnxBackend::OnnxBackend(OnnxBackend&&) noexcept = default;
OnnxBackend& OnnxBackend::operator=(OnnxBackend&&) noexcept = default;

std::string_view OnnxBackend::name() const noexcept {
  return impl_->name;
}
const TensorSpec& OnnxBackend::input_spec() const noexcept {
  return impl_->input_spec;
}
const TensorSpec& OnnxBackend::output_spec() const noexcept {
  return impl_->output_spec;
}

void OnnxBackend::infer(std::span<const float> input, std::span<float> output) {
  Impl& im = *impl_;
  if (input.size() < im.input_elements || output.size() < im.output_elements) {
    throw std::invalid_argument("OnnxBackend::infer: tensor size mismatch");
  }
  // Wrap the caller's buffers; OrtValue creation copies no tensor data. The C API takes mutable
  // pointers for inputs too, but never writes through them.
  Ort::Value in = Ort::Value::CreateTensor<float>(im.memory_info, const_cast<float*>(input.data()),
                                                  im.input_elements, im.input_spec.shape.data(),
                                                  im.input_spec.shape.size());
  Ort::Value out =
      Ort::Value::CreateTensor<float>(im.memory_info, output.data(), im.output_elements,
                                      im.output_spec.shape.data(), im.output_spec.shape.size());
  const char* input_name = im.input_spec.name.c_str();
  const char* output_name = im.output_spec.name.c_str();
  im.session.Run(im.run_options, &input_name, &in, 1, &output_name, &out, 1);
}

}  // namespace takt
