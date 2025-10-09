add_library(video_analysis_engine SHARED src/VideoAnalysisEngine.cpp)

target_link_libraries(video_analysis_engine PUBLIC
    common
    stream_handler
    frame_sampler
    object_detection
    sort_tracker
    clip_image_encoder
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
