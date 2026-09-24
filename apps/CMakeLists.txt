include(FetchContent)
FetchContent_Declare(cli11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.7.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(CLI11_PRECOMPILED OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cli11)

add_library(takt_app_support STATIC app_support.cpp)
target_include_directories(takt_app_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(takt_app_support PUBLIC takt_core)
if(TARGET takt_onnx)
  target_link_libraries(takt_app_support PUBLIC takt_onnx)
endif()
if(TARGET takt_opencv)
  target_link_libraries(takt_app_support PUBLIC takt_opencv)
endif()
takt_set_warnings(takt_app_support)

foreach(app takt_run takt_bench)
  add_executable(${app} ${app}.cpp)
  target_link_libraries(${app} PRIVATE takt_app_support CLI11::CLI11)
  takt_set_warnings(${app})
  if(WIN32 AND TARGET takt::onnxruntime)
    # Put onnxruntime.dll next to the executable so it runs from the build tree.
    add_custom_command(TARGET ${app} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:takt::onnxruntime> $<TARGET_FILE_DIR:${app}>)
  endif()
endforeach()
