#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "takt/vision/letterbox.hpp"

namespace takt {
namespace {

TEST(Letterbox, GeometryMatchesUltralytics) {
  // 1280x720 -> 640x640: scale 0.5, 640x360 content, 140 px bars top and bottom.
  const auto t = compute_letterbox(1280, 720, 640, 640);
  EXPECT_FLOAT_EQ(t.scale, 0.5f);
  EXPECT_EQ(t.resized_width, 640);
  EXPECT_EQ(t.resized_height, 360);
  EXPECT_FLOAT_EQ(t.pad_x, 0.0f);
  EXPECT_FLOAT_EQ(t.pad_y, 140.0f);

  // Ultralytics' bus.jpg, 810x1080: scale 640/1080, 480x640 content, 80 px bars left and right.
  const auto bus = compute_letterbox(810, 1080, 640, 640);
  EXPECT_EQ(bus.resized_width, 480);
  EXPECT_EQ(bus.resized_height, 640);
  EXPECT_FLOAT_EQ(bus.pad_x, 80.0f);
  EXPECT_FLOAT_EQ(bus.pad_y, 0.0f);

  // Odd padding: round(dw - 0.1) puts the extra pixel on the far side.
  const auto odd = compute_letterbox(641, 100, 640, 640);
  EXPECT_EQ(odd.resized_width, 640);
  EXPECT_FLOAT_EQ(odd.pad_y,
                  std::round(static_cast<float>(640 - odd.resized_height) / 2.0f - 0.1f));
}

TEST(Letterbox, RejectsInvalidSizes) {
  EXPECT_THROW(static_cast<void>(compute_letterbox(0, 10, 640, 640)), std::invalid_argument);
  EXPECT_THROW(Letterboxer(0, 640), std::invalid_argument);
}

TEST(Letterbox, BoxRoundTrip) {
  const auto t = compute_letterbox(1920, 1080, 640, 640);
  const Box source{100.0f, 200.0f, 500.0f, 900.0f};
  const Box back = t.to_source(t.to_model(source));
  EXPECT_NEAR(back.x1, source.x1, 1e-3);
  EXPECT_NEAR(back.y1, source.y1, 1e-3);
  EXPECT_NEAR(back.x2, source.x2, 1e-3);
  EXPECT_NEAR(back.y2, source.y2, 1e-3);
}

TEST(Letterbox, ToSourceClipsToImage) {
  const auto t = compute_letterbox(1280, 720, 640, 640);
  const Box clipped = t.to_source({-50.0f, 0.0f, 700.0f, 640.0f});
  EXPECT_FLOAT_EQ(clipped.x1, 0.0f);
  EXPECT_FLOAT_EQ(clipped.y1, 0.0f);
  EXPECT_FLOAT_EQ(clipped.x2, 1280.0f);
  EXPECT_FLOAT_EQ(clipped.y2, 720.0f);
}

TEST(Letterbox, SameSizeIsExactColourConversionAndNormalisation) {
  Image image(4, 4, PixelFormat::kBgr8);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      std::uint8_t* px = image.row(y) + x * 3;
      px[0] = static_cast<std::uint8_t>(10 * x);       // B
      px[1] = static_cast<std::uint8_t>(20 * y);       // G
      px[2] = static_cast<std::uint8_t>(200 + x + y);  // R
    }
  }
  Letterboxer lb(4, 4);
  std::vector<float> chw(lb.tensor_size());
  const auto t = lb.run(image.view(), chw);
  EXPECT_FLOAT_EQ(t.scale, 1.0f);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      const auto i = static_cast<std::size_t>(y * 4 + x);
      EXPECT_FLOAT_EQ(chw[i], static_cast<float>(200 + x + y) / 255.0f) << "R plane";
      EXPECT_FLOAT_EQ(chw[16 + i], static_cast<float>(20 * y) / 255.0f) << "G plane";
      EXPECT_FLOAT_EQ(chw[32 + i], static_cast<float>(10 * x) / 255.0f) << "B plane";
    }
  }
}

TEST(Letterbox, PaddingUsesGrey114) {
  Image image(8, 4, PixelFormat::kBgr8);  // wide image -> bars top and bottom
  std::fill(image.data(), image.data() + image.size_bytes(), std::uint8_t{255});
  Letterboxer lb(8, 8);
  std::vector<float> chw(lb.tensor_size(), -1.0f);
  const auto t = lb.run(image.view(), chw);
  EXPECT_FLOAT_EQ(t.pad_y, 2.0f);
  for (std::size_t p = 0; p < 3; ++p) {
    const float* plane = chw.data() + p * 64;
    for (int x = 0; x < 8; ++x) {
      EXPECT_FLOAT_EQ(plane[x], 114.0f / 255.0f);          // row 0: padding
      EXPECT_FLOAT_EQ(plane[3 * 8 + x], 1.0f);             // row 3: image
      EXPECT_FLOAT_EQ(plane[7 * 8 + x], 114.0f / 255.0f);  // row 7: padding
    }
  }
}

TEST(Letterbox, DownscaleAveragesNeighbours) {
  // 2x downscale with half-pixel centres samples exactly between two source pixels.
  Image image(4, 1, PixelFormat::kGray8);
  image.row(0)[0] = 0;
  image.row(0)[1] = 100;
  image.row(0)[2] = 200;
  image.row(0)[3] = 250;
  Letterboxer lb(2, 2);
  std::vector<float> chw(lb.tensor_size());
  const auto t = lb.run(image.view(), chw);
  ASSERT_EQ(t.resized_width, 2);
  ASSERT_EQ(t.resized_height, 1);
  const auto row = static_cast<std::size_t>(t.pad_y) * 2;
  EXPECT_NEAR(chw[row + 0], 50.0f / 255.0f, 1e-6);
  EXPECT_NEAR(chw[row + 1], 225.0f / 255.0f, 1e-6);
  EXPECT_FLOAT_EQ(chw[row + 0], chw[4 + row + 0]) << "grey replicates into all three planes";
}

TEST(Letterbox, HonoursRowStride) {
  // A view with padded rows (as delivered by many camera drivers) must give the same result.
  Image packed(3, 2, PixelFormat::kBgr8);
  std::vector<std::uint8_t> padded(2 * 16, 0);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 9; ++x) {
      const auto v = static_cast<std::uint8_t>(y * 50 + x * 7);
      packed.row(y)[x] = v;
      padded[static_cast<std::size_t>(y * 16 + x)] = v;
    }
  }
  const ImageView strided{padded.data(), 3, 2, 16, PixelFormat::kBgr8};
  Letterboxer lb(6, 6);
  std::vector<float> a(lb.tensor_size());
  std::vector<float> b(lb.tensor_size());
  static_cast<void>(lb.run(packed.view(), a));
  static_cast<void>(lb.run(strided, b));
  EXPECT_EQ(a, b);
}

TEST(Letterbox, RejectsTooSmallOutput) {
  Image image(4, 4, PixelFormat::kBgr8);
  Letterboxer lb(4, 4);
  std::vector<float> small(10);
  EXPECT_THROW(static_cast<void>(lb.run(image.view(), small)), std::invalid_argument);
}

}  // namespace
}  // namespace takt
