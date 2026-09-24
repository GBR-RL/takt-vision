#include "takt/track/linear_assignment.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace takt {

void LinearAssignment::solve(std::span<const float> cost, int rows, int cols, float cost_limit,
                             AssignmentResult& result) {
  result.clear();
  if (rows < 0 || cols < 0 ||
      cost.size() < static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
    throw std::invalid_argument("LinearAssignment::solve: cost matrix size mismatch");
  }
  if (rows == 0 || cols == 0) {
    for (int r = 0; r < rows; ++r) result.unmatched_rows.push_back(r);
    for (int c = 0; c < cols; ++c) result.unmatched_cols.push_back(c);
    return;
  }

  // Extended square problem of size n = rows + cols:
  //   [ C (gated)        | L/2 on diagonal ]
  //   [ L/2 on diagonal  | 0               ]
  const int n = rows + cols;
  const auto un = static_cast<std::size_t>(n);
  const std::size_t stride = un + 1;
  constexpr double kForbidden = 1e9;
  const double half_limit = 0.5 * static_cast<double>(cost_limit);

  matrix_.assign(stride * stride, 0.0);
  const auto at = [&](int r, int c) -> double& {  // 0-indexed accessor into the 1-indexed buffer
    return matrix_[static_cast<std::size_t>(r + 1) * stride + static_cast<std::size_t>(c + 1)];
  };
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      double value = 0.0;
      if (r < rows && c < cols) {
        const float cst = cost[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                               static_cast<std::size_t>(c)];
        value = cst <= cost_limit ? static_cast<double>(cst) : kForbidden;
      } else if (r < rows) {
        value = (c - cols == r) ? half_limit : kForbidden;  // row r -> its own dummy column
      } else if (c < cols) {
        value = (r - rows == c) ? half_limit : kForbidden;  // column c -> its own dummy row
      }
      at(r, c) = value;
    }
  }

  // Hungarian algorithm with potentials (shortest augmenting paths), O(n^3).
  constexpr double kInf = std::numeric_limits<double>::infinity();
  u_.assign(stride, 0.0);
  v_.assign(stride, 0.0);
  assigned_row_.assign(stride, 0);
  way_.assign(stride, 0);
  for (int i = 1; i <= n; ++i) {
    assigned_row_[0] = i;
    std::size_t j0 = 0;
    min_slack_.assign(stride, kInf);
    used_.assign(stride, 0);
    do {
      used_[j0] = 1;
      const auto i0 = static_cast<std::size_t>(assigned_row_[j0]);
      double delta = kInf;
      std::size_t j1 = 0;
      for (std::size_t j = 1; j <= un; ++j) {
        if (used_[j] != 0) continue;
        const double reduced = matrix_[i0 * stride + j] - u_[i0] - v_[j];
        if (reduced < min_slack_[j]) {
          min_slack_[j] = reduced;
          way_[j] = static_cast<int>(j0);
        }
        if (min_slack_[j] < delta) {
          delta = min_slack_[j];
          j1 = j;
        }
      }
      for (std::size_t j = 0; j <= un; ++j) {
        if (used_[j] != 0) {
          u_[static_cast<std::size_t>(assigned_row_[j])] += delta;
          v_[j] -= delta;
        } else {
          min_slack_[j] -= delta;
        }
      }
      j0 = j1;
    } while (assigned_row_[j0] != 0);
    do {  // augment along the alternating path
      const auto j1 = static_cast<std::size_t>(way_[j0]);
      assigned_row_[j0] = assigned_row_[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  // Read back the real (row, col) pairs; anything assigned to a dummy is unmatched.
  std::vector<char>& row_matched = used_;  // reuse scratch: indices 0..rows-1
  row_matched.assign(static_cast<std::size_t>(rows), 0);
  for (int c = 0; c < cols; ++c) {
    const int r = assigned_row_[static_cast<std::size_t>(c + 1)] - 1;
    if (r >= 0 && r < rows && at(r, c) < kForbidden) {
      result.matches.emplace_back(r, c);
      row_matched[static_cast<std::size_t>(r)] = 1;
    } else {
      result.unmatched_cols.push_back(c);
    }
  }
  for (int r = 0; r < rows; ++r) {
    if (row_matched[static_cast<std::size_t>(r)] == 0) result.unmatched_rows.push_back(r);
  }
}

}  // namespace takt
