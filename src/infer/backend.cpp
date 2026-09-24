#include "takt/infer/backend.hpp"

#include <stdexcept>

namespace takt {

std::size_t TensorSpec::element_count() const {
  std::size_t count = 1;
  for (const std::int64_t dim : shape) {
    if (dim <= 0) throw std::logic_error("TensorSpec '" + name + "' has a dynamic dimension");
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

}  // namespace takt
