cmake_minimum_required(VERSION 3.25)

project(takt_vision
  VERSION 0.1.0
  DESCRIPTION "Real-time C++20 vision inference pipeline"
  HOMEPAGE_URL "https://github.com/GBR-RL/takt-vision"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TAKT_BUILD_APPS "Build the takt_run / takt_bench command-line tools" ${PROJECT_IS_TOP_LEVEL})
option(TAKT_BUILD_TESTS "Build unit and integration tests" ${PROJECT_IS_TOP_LEVEL})
option(TAKT_BUILD_BENCHMARKS "Build Google Benchmark micro-benchmarks" OFF)
option(TAKT_WITH_ONNXRUNTIME "Build the ONNX Runtime inference backend" ON)
option(TAKT_FETCH_ONNXRUNTIME "Download a prebuilt ONNX Runtime when ONNXRUNTIME_ROOT is not set" ON)
option(TAKT_WITH_OPENCV "Build OpenCV video sources and overlay sinks" ON)
option(TAKT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(TAKT_SANITIZER "" CACHE STRING "Sanitizers to enable: 'address;undefined', 'thread', or empty")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

find_package(Threads REQUIRED)

# --- takt_core: the pipeline itself. Depends on nothing but the standard library. -------------
add_library(takt_core
  src/core/image.cpp
  src/core/latency_histogram.cpp
  src/vision/letterbox.cpp
  src/vision/nms.cpp
  src/vision/yolo_decoder.cpp
  src/track/byte_tracker.cpp
  src/track/kalman_filter.cpp
  src/track/linear_assignment.cpp
  src/infer/backend.cpp
  src/infer/fake_backend.cpp
  src/pipeline/frame_pool.cpp
  src/pipeline/hand_off.cpp
  src/pipeline/metrics.cpp
  src/pipeline/pipeline.cpp
  src/io/json_lines_sink.cpp
  src/io/synthetic_source.cpp)
add_library(takt::core ALIAS takt_core)
target_include_directories(takt_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(takt_core PUBLIC Threads::Threads)
target_compile_features(takt_core PUBLIC cxx_std_20)
takt_set_warnings(takt_core)

# --- takt_onnx: ONNX Runtime backend (CPU by default, CUDA execution provider if available). ---
if(TAKT_WITH_ONNXRUNTIME)
  include(OnnxRuntime)
  add_library(takt_onnx src/infer/onnx_backend.cpp)
  add_library(takt::onnx ALIAS takt_onnx)
  target_link_libraries(takt_onnx PUBLIC takt_core PRIVATE takt::onnxruntime)
  target_compile_definitions(takt_onnx PUBLIC TAKT_HAS_ONNXRUNTIME=1)
  takt_set_warnings(takt_onnx)
endif()

# --- takt_opencv: camera / video sources, overlay rendering, video and window sinks. ----------
if(TAKT_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio highgui)
  if(OpenCV_FOUND)
    add_library(takt_opencv
      src/io/opencv_source.cpp
      src/io/overlay.cpp
      src/io/overlay_sink.cpp)
    add_library(takt::opencv ALIAS takt_opencv)
    target_link_libraries(takt_opencv PUBLIC takt_core ${OpenCV_LIBS})
    target_include_directories(takt_opencv SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(takt_opencv PUBLIC TAKT_HAS_OPENCV=1)
    takt_set_warnings(takt_opencv)
    message(STATUS "takt: OpenCV ${OpenCV_VERSION} found - video I/O enabled")
  else()
    message(WARNING "takt: OpenCV not found - building without video I/O (synthetic source only)")
  endif()
endif()

if(TAKT_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(TAKT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TAKT_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
