add_library(storage_handler SHARED src/milvus_storage_handler.cpp)

target_include_directories(storage_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(storage_handler PRIVATE
    milvus_sdk
    common
    pthread
)
