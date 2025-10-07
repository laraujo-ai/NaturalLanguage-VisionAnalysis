cmake_minimum_required(VERSION 3.16)
project(GStreamerRTSPDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${GST_APP_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/vision_stream_handlers.cpp
    ../../common/include/interfaces.hpp
    demo.cpp
)

# Create executable
add_executable(rtsp_demo ${SOURCES})

add_library(stream_handler SHARED src/vision_stream_handlers.cpp ../../common/include/interfaces.hpp)

# Link libraries
target_link_libraries(rtsp_demo
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
    pthread
)
target_link_libraries(stream_handler
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
    pthread
)

# Compiler flags
target_compile_options(rtsp_demo PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GST_APP_CFLAGS_OTHER}
)

# Add include paths
target_include_directories(rtsp_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

message(STATUS "GStreamer found: ${GSTREAMER_FOUND}")
message(STATUS "GStreamer includes: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "GStreamer libraries: ${GSTREAMER_LIBRARIES}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")