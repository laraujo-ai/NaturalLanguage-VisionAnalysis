find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

add_library(media_processing SHARED src/MediaProcessor.cpp)

target_link_libraries(media_processing PUBLIC
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
    pthread
)

target_include_directories(media_processing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
)

target_compile_options(media_processing PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GST_APP_CFLAGS_OTHER}
)

option(BUILD_STREAM_HANDLER_DEMO "Build stream handler demo" OFF)
if(BUILD_STREAM_HANDLER_DEMO)
    add_executable(stream_handler_demo demo.cpp)
    target_link_libraries(stream_handler_demo PRIVATE media_processing)
    target_include_directories(stream_handler_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    )
endif()
