cmake_minimum_required(VERSION 3.16)
project(StreamProcessing-NaturalLanguageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
        ${CMAKE_SOURCE_DIR}/../onnxruntime/lib
        ${CMAKE_SOURCE_DIR}/onnxruntime/lib
        /usr/local/lib
        /opt/onnxruntime/lib
)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
        ${CMAKE_SOURCE_DIR}/../onnxruntime/include
        ${CMAKE_SOURCE_DIR}/onnxruntime/include
        /usr/local/include/onnxruntime
        /opt/onnxruntime/include
)

if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime not found. Please run install_onnxruntime.sh first.")
endif()

message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
)

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
message(STATUS "Eigen3 include dirs: ${EIGEN3_INCLUDE_DIRS}")

find_package(nlohmann_json REQUIRED)
message(STATUS "nlohmann_json found")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

message(STATUS "GStreamer found: ${GSTREAMER_FOUND}")
message(STATUS "GStreamer include dirs: ${GSTREAMER_INCLUDE_DIRS}")

include_directories(
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
)

add_subdirectory(src)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
    video_analysis_engine
    stream_handler
    frame_sampler
    common
    object_detection
    onnxruntime
    sort_tracker
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GSTREAMER_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
)

