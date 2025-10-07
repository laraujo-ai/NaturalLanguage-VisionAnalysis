find_package(OpenCV REQUIRED)

# Create common library (header-only or with source files)
add_library(common INTERFACE)

# Interface include directories
target_include_directories(common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Interface link libraries
target_link_libraries(common INTERFACE
    ${OpenCV_LIBS}
)
