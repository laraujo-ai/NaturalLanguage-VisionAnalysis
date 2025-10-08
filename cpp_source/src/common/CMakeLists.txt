add_library(common SHARED
    src/config_parser.cpp
)

target_link_libraries(common PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
    Eigen3::Eigen
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
