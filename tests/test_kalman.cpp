#include <gtest/gtest.h>

#include <cmath>

#include "takt/track/kalman_filter.hpp"

namespace takt {
namespace {

TEST(BoxKalmanFilter, MeasurementRoundTrip) {
  const Box box{10.0f, 20.0f, 50.0f, 100.0f};
  const auto z = BoxKalmanFilter::to_measurement(box);
  EXPECT_FLOAT_EQ(z(0, 0), 30.0f);
  EXPECT_FLOAT_EQ(z(1, 0), 60.0f);
  EXPECT_FLOAT_EQ(z(2, 0), 0.5f);
  EXPECT_FLOAT_EQ(z(3, 0), 80.0f);
  const auto state = BoxKalmanFilter::initiate(z);
  const Box back = BoxKalmanFilter::to_box(state.mean);
  EXPECT_NEAR(back.x1, box.x1, 1e-4);
  EXPECT_NEAR(back.y2, box.y2, 1e-4);
}

TEST(BoxKalmanFilter, LearnsConstantVelocity) {
  // Object moving +4 px/frame in x, +2 px/frame in y.
  auto state = BoxKalmanFilter::initiate(BoxKalmanFilter::to_measurement({0, 0, 40, 80}));
  for (int t = 1; t <= 60; ++t) {
    BoxKalmanFilter::predict(state);
    const auto dx = 4.0f * static_cast<float>(t);
    const auto dy = 2.0f * static_cast<float>(t);
    BoxKalmanFilter::update(state, BoxKalmanFilter::to_measurement({dx, dy, dx + 40, dy + 80}));
  }
  EXPECT_NEAR(state.mean(4, 0), 4.0f, 0.05f);
  EXPECT_NEAR(state.mean(5, 0), 2.0f, 0.05f);

  // Prediction without a measurement extrapolates the motion.
  BoxKalmanFilter::predict(state);
  const Box predicted = BoxKalmanFilter::to_box(state.mean);
  EXPECT_NEAR(predicted.x1, 4.0f * 61, 0.5f);
  EXPECT_NEAR(predicted.y1, 2.0f * 61, 0.5f);
}

TEST(BoxKalmanFilter, CovarianceStaysSymmetricAndShrinksWithMeasurements) {
  auto state = BoxKalmanFilter::initiate(BoxKalmanFilter::to_measurement({0, 0, 40, 80}));
  const float initial_var = state.covariance(0, 0);
  for (int t = 0; t < 20; ++t) {
    BoxKalmanFilter::predict(state);
    BoxKalmanFilter::update(state, BoxKalmanFilter::to_measurement({0, 0, 40, 80}));
  }
  for (std::size_t r = 0; r < 8; ++r) {
    for (std::size_t c = 0; c < 8; ++c) {
      EXPECT_NEAR(state.covariance(r, c), state.covariance(c, r),
                  1e-3f * (1.0f + std::fabs(state.covariance(r, c))));
    }
  }
  EXPECT_LT(state.covariance(0, 0), initial_var);
}

}  // namespace
}  // namespace takt
