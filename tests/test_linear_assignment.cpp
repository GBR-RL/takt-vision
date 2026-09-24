#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "takt/track/linear_assignment.hpp"

namespace takt {
namespace {

// Reference: exhaustive search over every partial matching, with the same gating semantics as
// lap.lapjv(extend_cost=True, cost_limit): each unmatched row or column costs limit / 2.
double brute_force_cost(const std::vector<float>& cost, int rows, int cols, float limit) {
  double best = 1e18;
  std::vector<char> used(static_cast<std::size_t>(cols), 0);
  const auto recurse = [&](auto&& self, int row, double acc, int matched) -> void {
    if (row == rows) {
      const double unmatched = (rows - matched) + (cols - matched);
      best = std::min(best, acc + unmatched * static_cast<double>(limit) / 2.0);
      return;
    }
    self(self, row + 1, acc, matched);  // leave this row unmatched
    for (int c = 0; c < cols; ++c) {
      const float v = cost[static_cast<std::size_t>(row * cols + c)];
      if (used[static_cast<std::size_t>(c)] != 0 || v > limit) continue;
      used[static_cast<std::size_t>(c)] = 1;
      self(self, row + 1, acc + static_cast<double>(v), matched + 1);
      used[static_cast<std::size_t>(c)] = 0;
    }
  };
  recurse(recurse, 0, 0.0, 0);
  return best;
}

double solution_cost(const AssignmentResult& r, const std::vector<float>& cost, int rows, int cols,
                     float limit) {
  double total = 0.0;
  for (const auto& [row, col] : r.matches)
    total += static_cast<double>(cost[static_cast<std::size_t>(row * cols + col)]);
  const auto unmatched = static_cast<double>(r.unmatched_rows.size() + r.unmatched_cols.size());
  EXPECT_EQ(r.matches.size() + r.unmatched_rows.size(), static_cast<std::size_t>(rows));
  EXPECT_EQ(r.matches.size() + r.unmatched_cols.size(), static_cast<std::size_t>(cols));
  return total + unmatched * static_cast<double>(limit) / 2.0;
}

TEST(LinearAssignment, SolvesSquareProblem) {
  // Minimum total is 6 (e.g. row0->col1, row1->col0, row2->col2 = 1 + 2 + 3).
  const std::vector<float> cost = {4, 1, 3, 2, 0, 5, 3, 2, 3};
  LinearAssignment solver;
  AssignmentResult result;
  solver.solve(cost, 3, 3, 100.0f, result);
  ASSERT_EQ(result.matches.size(), 3u);
  EXPECT_DOUBLE_EQ(solution_cost(result, cost, 3, 3, 100.0f), 6.0);
}

TEST(LinearAssignment, GatingLeavesExpensivePairsUnmatched) {
  const std::vector<float> cost = {0.1f, 0.95f, 0.9f, 0.2f};
  LinearAssignment solver;
  AssignmentResult result;
  solver.solve(cost, 2, 2, 0.8f, result);
  ASSERT_EQ(result.matches.size(), 2u);
  for (const auto& [r, c] : result.matches) EXPECT_EQ(r, c);

  const std::vector<float> all_bad = {0.9f, 0.95f, 0.99f, 0.85f};
  solver.solve(all_bad, 2, 2, 0.8f, result);
  EXPECT_TRUE(result.matches.empty());
  EXPECT_EQ(result.unmatched_rows.size(), 2u);
  EXPECT_EQ(result.unmatched_cols.size(), 2u);
}

TEST(LinearAssignment, HandlesEmptyDimensions) {
  LinearAssignment solver;
  AssignmentResult result;
  solver.solve({}, 0, 3, 0.8f, result);
  EXPECT_EQ(result.unmatched_cols.size(), 3u);
  solver.solve({}, 2, 0, 0.8f, result);
  EXPECT_EQ(result.unmatched_rows.size(), 2u);
}

TEST(LinearAssignment, MatchesBruteForceOnRandomRectangularProblems) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  LinearAssignment solver;
  AssignmentResult result;
  for (int trial = 0; trial < 300; ++trial) {
    const int rows = 1 + static_cast<int>(rng() % 5);
    const int cols = 1 + static_cast<int>(rng() % 5);
    std::vector<float> cost(static_cast<std::size_t>(rows * cols));
    for (auto& v : cost) v = uniform(rng);
    const float limit = 0.3f + 0.6f * uniform(rng);
    solver.solve(cost, rows, cols, limit, result);
    ASSERT_NEAR(solution_cost(result, cost, rows, cols, limit),
                brute_force_cost(cost, rows, cols, limit), 1e-5)
        << "trial " << trial << " (" << rows << "x" << cols << ", limit " << limit << ")";
  }
}

}  // namespace
}  // namespace takt
