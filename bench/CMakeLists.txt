include(FetchContent)
FetchContent_Declare(benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(takt_microbench microbench.cpp)
target_link_libraries(takt_microbench PRIVATE takt_core benchmark::benchmark_main)
takt_set_warnings(takt_microbench)
