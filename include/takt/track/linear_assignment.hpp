#pragma once

#include <span>
#include <utility>
#include <vector>

namespace takt {

struct AssignmentResult {
  std::vector<std::pair<int, int>> matches;  // (row, col)
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_cols;

  void clear() noexcept {
    matches.clear();
    unmatched_rows.clear();
    unmatched_cols.clear();
  }
};

// Optimal rectangular assignment (Hungarian algorithm, O(n^3)) with a per-pair cost limit.
//
// Gating is done the way lap.lapjv(extend_cost=True, cost_limit=L) does it, which is what the
// reference ByteTrack uses: the matrix is extended so every row and every column may instead be
// left unmatched at cost L/2. A real pair is therefore chosen only if it is cheaper than leaving
// both of its ends unmatched, and the result is optimal *under* the gate - unlike the common
// shortcut of solving first and discarding expensive pairs afterwards.
class LinearAssignment {
 public:
  // `cost` is row-major, rows x cols.
  void solve(std::span<const float> cost, int rows, int cols, float cost_limit,
             AssignmentResult& result);

 private:
  std::vector<double> matrix_;  // (n + 1) x (n + 1), 1-indexed as in the classic formulation
  std::vector<double> u_;
  std::vector<double> v_;
  std::vector<double> min_slack_;
  std::vector<int> assigned_row_;  // column j -> row assigned to it (0 = none)
  std::vector<int> way_;
  std::vector<char> used_;
};

}  // namespace takt
