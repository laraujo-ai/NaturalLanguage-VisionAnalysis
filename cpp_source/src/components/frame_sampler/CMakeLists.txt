add_library(frame_sampler SHARED src/frame_samplers.cpp)

target_link_libraries(frame_sampler PUBLIC
    common
    ${OpenCV_LIBS}
    pthread
)

target_include_directories(frame_sampler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

option(BUILD_FRAME_SAMPLER_TESTS "Build frame sampler tests" ON)
if(BUILD_FRAME_SAMPLER_TESTS)
    add_subdirectory(tests)
endif()
