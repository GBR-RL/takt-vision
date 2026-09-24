#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace takt {

struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;

  // Product of all dimensions. Throws if any dimension is still dynamic (<= 0).
  [[nodiscard]] std::size_t element_count() const;
};

// What the pipeline needs from an inference engine: fixed float32 input/output tensors and a
// synchronous infer(). Engines satisfy this structurally - no base class, no virtual functions in
// their own code - and stay individually testable.
template <class B>
concept InferenceBackend =
    std::movable<B> && requires(B& backend, const B& const_backend, std::span<const float> input,
                                std::span<float> output) {
      { const_backend.name() } -> std::convertible_to<std::string_view>;
      { const_backend.input_spec() } -> std::same_as<const TensorSpec&>;
      { const_backend.output_spec() } -> std::same_as<const TensorSpec&>;
      { backend.infer(input, output) } -> std::same_as<void>;
    };

// Type-erased owner of any InferenceBackend ("external polymorphism"): the pipeline can pick
// ONNX Runtime, TensorRT or the fake backend at runtime from a command-line flag, while each
// engine is still an ordinary value type checked by the concept at compile time. The single
// virtual call per frame is noise next to a multi-millisecond forward pass.
class Backend {
 public:
  template <class B>
    requires(!std::same_as<std::remove_cvref_t<B>, Backend> &&
             InferenceBackend<std::remove_cvref_t<B>>)
  Backend(B&& backend)  // NOLINT(google-explicit-constructor): implicit wrapping is the point
      : self_(std::make_unique<Model<std::remove_cvref_t<B>>>(std::forward<B>(backend))) {}

  Backend(Backend&&) noexcept = default;
  Backend& operator=(Backend&&) noexcept = default;
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  ~Backend() = default;

  [[nodiscard]] std::string_view name() const { return self_->name(); }
  [[nodiscard]] const TensorSpec& input_spec() const { return self_->input_spec(); }
  [[nodiscard]] const TensorSpec& output_spec() const { return self_->output_spec(); }
  void infer(std::span<const float> input, std::span<float> output) { self_->infer(input, output); }

 private:
  struct Concept {
    Concept() = default;
    Concept(const Concept&) = delete;
    Concept& operator=(const Concept&) = delete;
    Concept(Concept&&) = delete;
    Concept& operator=(Concept&&) = delete;
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual const TensorSpec& input_spec() const = 0;
    [[nodiscard]] virtual const TensorSpec& output_spec() const = 0;
    virtual void infer(std::span<const float> input, std::span<float> output) = 0;
  };

  template <class B>
  struct Model final : Concept {
    template <class Arg>
    explicit Model(Arg&& arg) : impl(std::forward<Arg>(arg)) {}
    [[nodiscard]] std::string_view name() const override { return impl.name(); }
    [[nodiscard]] const TensorSpec& input_spec() const override { return impl.input_spec(); }
    [[nodiscard]] const TensorSpec& output_spec() const override { return impl.output_spec(); }
    void infer(std::span<const float> input, std::span<float> output) override {
      impl.infer(input, output);
    }
    B impl;
  };

  std::unique_ptr<Concept> self_;
};

}  // namespace takt
