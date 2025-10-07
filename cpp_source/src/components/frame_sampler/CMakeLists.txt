find_package(OpenCV REQUIRED)

# Create library
add_library(frame_sampler SHARED src/frame_samplers.cpp)

# Link libraries
target_link_libraries(frame_sampler PUBLIC
    ${OpenCV_LIBS}
    pthread
)

# Include directories
target_include_directories(frame_sampler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    ${OpenCV_INCLUDE_DIRS}
)

# Optional: Build demo executable (can be disabled)
option(BUILD_FRAME_SAMPLER_DEMO "Build frame sampler demo" OFF)
if(BUILD_FRAME_SAMPLER_DEMO)
    add_executable(frame_sampler_demo demo.cpp)
    target_link_libraries(frame_sampler_demo PRIVATE frame_sampler)
    target_include_directories(frame_sampler_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    )
endif()