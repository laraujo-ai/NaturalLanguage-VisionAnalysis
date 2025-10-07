cmake_minimum_required(VERSION 3.16)

project(FrameSampler)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(frame_sampler SHARED src/frame_samplers.cpp ../../common/include/interfaces.hpp)
add_executable(${PROJECT_NAME}_demo demo.cpp ../../common/include/interfaces.hpp)

target_link_libraries(frame_sampler PRIVATE ${OpenCV_LIBS}
pthread)