add_library(vlm_engine SHARED
    src/clip_image_encoder.cpp
    src/clip_text_encoder.cpp
)

target_include_directories(vlm_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(vlm_engine
    PRIVATE
        onnxruntime
        common
        ${OpenCV_LIBS}

)