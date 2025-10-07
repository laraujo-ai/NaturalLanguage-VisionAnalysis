cmake_minimum_required(VERSION 3.16)
project(StreamProcessing-NaturalLanguageRetrieval)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Add subdirectories first to build libraries
add_subdirectory(src)

# Create main executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link against libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    media_processing
    stream_handler
    frame_sampler
    common
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
)

# Include directories for main
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GSTREAMER_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}

)

