cmake_minimum_required(VERSION 3.10)
project(StreamProcessing-NaturalLanguageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


add_executable(${PROJECT_NAME} src/main.cpp)
add_subdirectory(src)

target_link_libraries(${PROJECT_NAME} PRIVATE frame_sampler common stream_handler)

