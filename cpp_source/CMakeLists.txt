cmake_minimum_required(VERSION 3.16)
project(StreamProcessing-NaturalLanguageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Add subdirectories first to build libraries
add_subdirectory(src)

# Create main executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link against libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    stream_handler
    frame_sampler
    common
)

# Include directories for main
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/components/stream_handler/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/components/frame_sampler/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/include
)

