add_executable(test_sampler
    test_sampler.cpp
    ../../../../lib/catch2/catch_amalgamated.cpp
)

target_include_directories(test_sampler PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/include
    ${CMAKE_SOURCE_DIR}/src/components/frame_sampler
    ${CMAKE_SOURCE_DIR}/lib/catch2
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_sampler
    frame_sampler
    ${OpenCV_LIBS}
)

enable_testing()
add_test(NAME FrameSamplerTests COMMAND test_sampler)
