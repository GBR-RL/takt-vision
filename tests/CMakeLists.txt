include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(takt_tests
  test_spsc_ring.cpp
  test_latest_slot.cpp
  test_latency_histogram.cpp
  test_letterbox.cpp
  test_yolo_decoder.cpp
  test_nms.cpp
  test_small_matrix.cpp
  test_kalman.cpp
  test_linear_assignment.cpp
  test_byte_tracker.cpp
  test_backend.cpp
  test_frame_pool.cpp
  test_pipeline.cpp)
target_link_libraries(takt_tests PRIVATE takt_core GTest::gtest_main)
takt_set_warnings(takt_tests)
gtest_discover_tests(takt_tests DISCOVERY_MODE PRE_TEST)

# Steady-state allocation check. Replaces global operator new, which does not mix with sanitizer
# runtimes, so it is only built without them.
if(NOT TAKT_SANITIZER)
  add_executable(takt_alloc_test test_zero_alloc.cpp)
  target_link_libraries(takt_alloc_test PRIVATE takt_core GTest::gtest)
  takt_set_warnings(takt_alloc_test)
  gtest_discover_tests(takt_alloc_test DISCOVERY_MODE PRE_TEST)
endif()

# End-to-end test of the ONNX Runtime backend against a tiny generated model with known outputs.
if(TARGET takt_onnx)
  find_package(Python3 COMPONENTS Interpreter)
  set(TAKT_TEST_MODEL "${CMAKE_CURRENT_BINARY_DIR}/tiny_yolo.onnx")
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/make_tiny_model.py ${TAKT_TEST_MODEL}
      RESULT_VARIABLE _tiny_model_result OUTPUT_QUIET ERROR_VARIABLE _tiny_model_error)
    if(NOT _tiny_model_result EQUAL 0)
      message(WARNING "takt: could not generate the test model (pip install onnx): ${_tiny_model_error}")
    endif()
  endif()
  add_executable(takt_onnx_tests test_onnx_backend.cpp)
  target_link_libraries(takt_onnx_tests PRIVATE takt_onnx GTest::gtest_main)
  target_compile_definitions(takt_onnx_tests PRIVATE TAKT_TEST_MODEL="${TAKT_TEST_MODEL}")
  takt_set_warnings(takt_onnx_tests)
  if(WIN32)
    add_custom_command(TARGET takt_onnx_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:takt::onnxruntime> $<TARGET_FILE_DIR:takt_onnx_tests>)
  endif()
  gtest_discover_tests(takt_onnx_tests DISCOVERY_MODE PRE_TEST)
endif()
