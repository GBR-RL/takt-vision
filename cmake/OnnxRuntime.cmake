# Provides the imported target takt::onnxruntime.
#
# Resolution order:
#   1. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>  (an extracted release package)
#   2. the ONNXRUNTIME_ROOT environment variable
#   3. download the official prebuilt release for this platform (TAKT_FETCH_ONNXRUNTIME=ON)
#
# Prebuilt packages exist for linux-x64, linux-aarch64, win-x64 and
# osx-arm64. For the CUDA execution provider, point ONNXRUNTIME_ROOT at a *-gpu_cuda12 package.

set(TAKT_ONNXRUNTIME_VERSION "1.30.0" CACHE STRING "ONNX Runtime release to download")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime release package")

if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(NOT ONNXRUNTIME_ROOT)
  if(NOT TAKT_FETCH_ONNXRUNTIME)
    message(FATAL_ERROR "takt: set ONNXRUNTIME_ROOT or enable TAKT_FETCH_ONNXRUNTIME")
  endif()

  set(_ort_version "${TAKT_ONNXRUNTIME_VERSION}")
  if(WIN32)
    set(_ort_package "onnxruntime-win-x64-${_ort_version}.zip")
  elseif(APPLE)
    set(_ort_package "onnxruntime-osx-arm64-${_ort_version}.tgz")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_ort_package "onnxruntime-linux-aarch64-${_ort_version}.tgz")
  else()
    set(_ort_package "onnxruntime-linux-x64-${_ort_version}.tgz")
  endif()

  include(FetchContent)
  FetchContent_Declare(onnxruntime_prebuilt
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_version}/${_ort_package}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(ONNXRUNTIME_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
  message(STATUS "takt: using downloaded ONNX Runtime ${_ort_version} (${_ort_package})")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  PATH_SUFFIXES "" onnxruntime onnxruntime/core/session
  NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "takt: ONNX Runtime headers or library not found under ${ONNXRUNTIME_ROOT}")
endif()

add_library(takt::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(takt::onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
if(WIN32)
  # find_library returns the import library; the DLL sits next to it in the release package.
  set_target_properties(takt::onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
else()
  set_target_properties(takt::onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}")
endif()
