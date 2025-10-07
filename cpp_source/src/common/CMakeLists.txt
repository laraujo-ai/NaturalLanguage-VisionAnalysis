find_package(OpenCV REQUIRED)

# Create common library with source files
add_library(common SHARED
    src/config_parser.cpp
    src/vision_factories.cpp
)

# Link libraries
target_link_libraries(common PUBLIC
    ${OpenCV_LIBS}
)

# Include directories
target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
