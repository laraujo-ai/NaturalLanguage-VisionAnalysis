add_library(video_analysis_engine SHARED src/VideoAnalysisEngine.cpp)

target_link_libraries(video_analysis_engine PUBLIC
    common
    stream_handler
    frame_sampler
    object_detection
    sort_tracker
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
    pthread
)

target_include_directories(video_analysis_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

target_compile_options(video_analysis_engine PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GST_APP_CFLAGS_OTHER}
)

# Optional demo build
option(BUILD_VIDEO_ANALYSIS_DEMO "Build video analysis engine demo" OFF)
if(BUILD_VIDEO_ANALYSIS_DEMO)
    add_executable(video_analysis_demo demo.cpp)
    target_link_libraries(video_analysis_demo PRIVATE video_analysis_engine)
    target_include_directories(video_analysis_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
    )
endif()
