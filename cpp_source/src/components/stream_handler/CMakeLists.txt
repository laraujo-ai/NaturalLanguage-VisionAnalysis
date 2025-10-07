# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

add_library(stream_handler SHARED src/vision_stream_handlers.cpp)

# Link libraries
target_link_libraries(stream_handler PUBLIC
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
    pthread
)

# Include directories
target_include_directories(stream_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
)

# Compiler flags
target_compile_options(stream_handler PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GST_APP_CFLAGS_OTHER}
)

# Optional: Build demo executable (can be disabled)
option(BUILD_STREAM_HANDLER_DEMO "Build stream handler demo" OFF)
if(BUILD_STREAM_HANDLER_DEMO)
    add_executable(stream_handler_demo demo.cpp)
    target_link_libraries(stream_handler_demo PRIVATE stream_handler)
    target_include_directories(stream_handler_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    )
endif()
